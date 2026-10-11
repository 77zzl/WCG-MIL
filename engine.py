import random
import zlib
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, recall_score, roc_auc_score)
from torch.nn import functional as F
from utils import save_json, save_checkpoint
_save_json_atomic = save_json
_save_checkpoint_atomic = save_checkpoint

def classification_metrics(labels, probabilities, num_classes, ordinal=False):
    
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("预测概率包含非有限值，禁止以替换概率或删除样本的方式计算指标。")
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes or np.any(probabilities < 0):
        raise FloatingPointError("预测概率形状或取值无效，禁止继续计算分类指标。")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise FloatingPointError("预测概率行和非正，禁止继续计算分类指标。")
    probabilities = probabilities / row_sums
    predictions = probabilities.argmax(1)
    class_ids = list(range(num_classes))
    confidence = probabilities.max(1)
    correctness = (predictions == labels).astype(float)
    calibration_error = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            calibration_error += selected.mean() * abs(
                correctness[selected].mean() - confidence[selected].mean()
            )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=class_ids, average="weighted", zero_division=0)),
        "balanced_accuracy": float(recall_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "per_class_f1": f1_score(labels, predictions, labels=class_ids, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(labels, predictions, labels=class_ids, average=None, zero_division=0).tolist(),
        "per_class_support": np.bincount(labels, minlength=num_classes).tolist(),
        "brier": float(np.mean(np.sum((probabilities - np.eye(num_classes)[labels]) ** 2, axis=1))),
        "ece_10": float(calibration_error),
    }
    if len(np.unique(labels)) == num_classes:
        if num_classes == 2:
            metrics["macro_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            metrics["macro_auc"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            )
            metrics["weighted_auc"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")
            )
    if ordinal:
        metrics["ordinal_mae"] = float(np.abs(predictions - labels).mean())
        metrics["quadratic_kappa"] = float(cohen_kappa_score(labels, predictions, weights="quadratic"))
        metrics["presence_f1"] = float(f1_score(labels > 0, predictions > 0, zero_division=0))
        metrics["presence_auc"] = float(roc_auc_score(labels > 0, 1 - probabilities[:, 0]))
    return metrics

def joint_nll(logits, labels, class_weights=None, training_priors=None):
    
    items = F.cross_entropy(logits.float(), labels, reduction='none')
    if class_weights is not None:
        weights = class_weights / (class_weights * training_priors).sum()
        items = items * weights[labels]
    return items.mean()

class NESTTrainer:
    

    def __init__(self, model, args, config, device, save_dir, log_dir):
        
        self.model, self.args, self.config, self.device = model, args, config, device
        self.save_dir, self.log_dir = Path(save_dir), Path(log_dir)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
        self.current_epoch = 0
        self.class_weights = None
        self.prototype_sources = {label: [] for label in range(config['num_classes'])}
        self.prototype_fallback, self.prototype_diagnostics = {}, {}
        self.resume_from = args.resume_training

    def _forward(self, inputs, mask, training):
        
        return self.model(inputs, mask)

    def _loss(self, output, labels):
        
        return joint_nll(output['logits'], labels, self.class_weights, self.training_priors)

    def _after_optimizer_step(self):
        
        pass

    def _prepare_training_protocol(self, train_loader):
        
        labels = np.asarray([record[1] for record in train_loader.dataset.records], dtype=np.int64)
        counts = np.bincount(labels, minlength=self.config['num_classes'])
        priors = counts / counts.sum()
        effective = (1 - self.args.effective_beta) / (1 - np.power(self.args.effective_beta, counts))
        effective = effective / np.sum(priors * effective)
        self.class_weights = torch.as_tensor(effective, dtype=torch.float32, device=self.device)
        self.training_priors = torch.as_tensor(priors, dtype=torch.float32, device=self.device)
        save_json({'class_counts': counts.tolist(), 'training_priors': priors.tolist(),
                   'normalized_effective_weights': effective.tolist(), 'loss': 'joint_nll'},
                  self.log_dir / 'loss_protocol.json')

    def _numerical_failure(self, reason, batch_index, sample_ids, inputs, output):
        
        record = {
            "reason": reason, "epoch": self.current_epoch, "batch": batch_index,
            "sample_ids": list(sample_ids), "input_shape": list(inputs.shape),
            "input_finite": bool(torch.isfinite(inputs).all()),
            "amp_scale": self.scaler.get_scale(),
            "nonfinite_outputs": [key for key, value in output.items()
                                  if torch.is_tensor(value) and not torch.isfinite(value).all()],
            "nonfinite_gradients": [name for name, parameter in self.model.named_parameters()
                                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()],
        }
        save_json(record, self.log_dir / "numerical_failure.json")
        raise FloatingPointError(f"{reason}；epoch={self.current_epoch}，batch={batch_index}，样本={sample_ids}")

    def _embed_record(self, path):
        
        features = torch.load(path, map_location="cpu", weights_only=True).float().to(self.device)
        mask = torch.ones(1, features.size(0), dtype=torch.bool, device=self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            return self.model(features.unsqueeze(0), mask)["prototype_features"][0].float().cpu()

    def _select_reservoir(self, train_dataset):
        
        by_class = {label: [] for label in range(self.config["num_classes"])}
        self.model.eval()
        for path, label, sample_id in train_dataset.records:
            features = torch.load(path, map_location="cpu", weights_only=True).float()
            mask = torch.ones(1, features.size(0), dtype=torch.bool, device=self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                output = self.model(features.unsqueeze(0).to(self.device), mask)
            risk = output["instance_logits"][0].float()
            reliability = output["reliability"][0].float()
            if reliability.ndim > 1:
                reliability = reliability.mean(-1)
            risk = risk + reliability.clamp_min(1e-6).log()
            if label > 0:
                probabilities = output["logits"].float().softmax(1)[0]
                eligible = int(probabilities.argmax()) == label and float(probabilities[label]) >= 0.7
                if eligible:
                    if getattr(self.model, "prototype_witness", "top") == "random":
                        gen = torch.Generator().manual_seed(self.args.seed + self.args.fold * 1009 + zlib.crc32(sample_id.encode()))
                        selected = torch.randperm(features.size(0), generator=gen)[:8]
                    else:
                        selected = risk.topk(min(8, features.size(0))).indices.cpu()
                    by_class[label].append((path, sample_id, selected.tolist()))
            else:
                selected = risk.topk(min(8, features.size(0)), largest=False).indices.cpu()
                by_class[0].append((path, sample_id, selected.tolist()))
        for label in range(1, self.config["num_classes"]):
            if not by_class[label]:
                by_class[label] = []
                for path, record_label, sample_id in train_dataset.records:
                    if record_label == label:
                        raw = torch.load(path, map_location="cpu", weights_only=True)
                        by_class[label].append((path, sample_id, list(range(min(8, raw.size(0))))))
                self.prototype_fallback[str(label)] = "全部训练袋回退"
        generator = torch.Generator().manual_seed(self.args.seed + self.args.fold)
        for label, records in by_class.items():
            flattened = [(path, sample_id, index) for path, sample_id, indices in records for index in indices]
            if len(flattened) > 512:
                selected_rows = torch.randperm(len(flattened), generator=generator)[:512].tolist()
                flattened = [flattened[index] for index in selected_rows]
            grouped = {}
            for path, sample_id, index in flattened:
                grouped.setdefault((str(path), sample_id), []).append(index)
            self.prototype_sources[label] = [
                {"path": path, "sample_id": sample_id, "indices": indices}
                for (path, sample_id), indices in grouped.items()
            ]
        self.model.train()

    def _refresh_prototypes(self, initial=False):
        
        vectors = {label: [] for label in range(self.config["num_classes"])}
        self.model.eval()
        for label, records in self.prototype_sources.items():
            for item in records:
                embedded = self._embed_record(item["path"])
                vectors[label].append(embedded[item["indices"]])
        prototype_rows = []
        for label in range(self.config["num_classes"]):
            samples = torch.cat(vectors[label], dim=0).numpy()
            clustering = MiniBatchKMeans(n_clusters=8, batch_size=min(1024, len(samples)),
                                         random_state=self.args.seed + self.args.fold + label, n_init=3)
            if initial:
                centers = clustering.fit(samples).cluster_centers_
            else:
                old = self.model.prototype_bank[label].detach().cpu().numpy()
                assignments = ((samples[:, None, :] - old[None, :, :]) ** 2).sum(-1).argmin(1)
                centers = old.copy()
                empty = []
                for cluster in range(8):
                    members = samples[assignments == cluster]
                    if len(members):
                        centers[cluster] = 0.9 * old[cluster] + 0.1 * members.mean(0)
                    else:
                        empty.append(cluster)
                self.prototype_fallback[f"empty_{label}"] = empty
            prototype_rows.append(torch.as_tensor(centers, dtype=torch.float32))
        self.model.set_prototypes(torch.stack(prototype_rows).to(self.device))
        self.model.train()
        self.prototype_diagnostics = {
            "ready": True, "initial": initial, "version_epoch": self.current_epoch,
            "reservoir_counts": {str(label): sum(len(item["indices"]) for item in records)
                                 for label, records in self.prototype_sources.items()},
            "fallback": self.prototype_fallback,
        }

    def _rng_state(self, train_loader):
        
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": train_loader.generator.get_state() if train_loader.generator is not None else None,
        }

    def _restore_rng(self, state, train_loader):
        
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        
        torch.set_rng_state(state["torch"].cpu())
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        if state["loader"] is not None:
            train_loader.generator.set_state(state["loader"].cpu())

    def _training_protocol(self):
        
        keys = ("name", "dataset", "fold", "seed", "feature_name", "split_protocol",
                "feature_bag_size", "optimizer", "epochs", "warmup_epochs", "patience",
                "lr", "weight_decay", "hidden_dim", "num_stages", "dropout",
                "class_balance", "effective_beta", "loss_mode", "presence_topk",
                "presence_temperature")
        protocol = {key: getattr(self.args, key, None) for key in keys}
        protocol.update(source_hash=getattr(self.args, "source_hash", None),
                        split_hash=getattr(self.args, "split_hash", None),
                        configuration=self.args.name)
        return protocol

    def fit(self, train_loader, val_loader):
        
        self._prepare_training_protocol(train_loader)
        history, best_score, best_epoch, stale_epochs = [], -1.0, 0, 0
        best_state, start_epoch = None, 1
        last_state_path = self.log_dir / "last_training_state.pt"
        protocol = self._training_protocol()
        if self.resume_from:
            state_path = Path(self.resume_from)
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            if state.get("protocol") == protocol:
                self.model.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.scaler.load_state_dict(state["scaler"])
                history, best_score, best_epoch = state["history"], state["best_score"], state["best_epoch"]
                stale_epochs, start_epoch = state["stale_epochs"], state["epoch"] + 1
                best_state, self.prototype_sources = state["best_state"], state["prototype_sources"]
                self.prototype_fallback = state["prototype_fallback"]
                self.prototype_diagnostics = state.get("prototype_diagnostics", {})
                self._restore_rng(state["rng"], train_loader)
                if self.args.save_weights and best_state is not None:
                    _save_checkpoint_atomic({"model": best_state, "epoch": best_epoch,
                                             "prototype_sources": self.prototype_sources,
                                             "prototype_fallback": self.prototype_fallback},
                                            self.save_dir / "best_model.pt")
            else:
                _save_json_atomic({"accepted": False, "checkpoint": str(state_path),
                                   "stored_protocol": state.get("protocol"),
                                   "current_protocol": protocol,
                                   "action": "保留旧状态；当前运行从头训练"},
                                  self.log_dir / "resume_rejected.json")
                print("断点状态与当前源码、划分或训练配方不一致，本run保留旧状态并从头训练。", flush=True)
        checkpoint = self.save_dir / "best_model.pt"
        for epoch in range(start_epoch, self.args.epochs + 1):
            self.current_epoch = epoch
            epoch_start = perf_counter()
            if epoch <= self.args.warmup_epochs:
                for group in self.optimizer.param_groups:
                    group["lr"] = self.args.lr * epoch / max(1, self.args.warmup_epochs)
            train_metrics, _, _, _, _ = self._run_epoch(train_loader, True)
            if self.model.reference and epoch > self.args.warmup_epochs:
                self._refresh_prototypes()
            train_seconds = perf_counter() - epoch_start
            val_metrics, _, _, _, _ = self._run_epoch(val_loader, False)
            val_seconds = perf_counter() - epoch_start - train_seconds
            if epoch > self.args.warmup_epochs:
                self.scheduler.step()
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                   "train_seconds": train_seconds, "val_seconds": val_seconds,
                   "epoch_seconds": perf_counter() - epoch_start,
                   "lr": self.optimizer.param_groups[0]["lr"]}
            history.append(row)
            _save_json_atomic(history, self.log_dir / "history.json")
            self.epoch_diagnostics = {"epoch": epoch, "configuration": self.args.name}
            if self.model.reference and bool(self.model.prototype_ready):
                self.epoch_diagnostics["prototype_update"] = self.prototype_diagnostics
                self.epoch_diagnostics["prototype_scale"] = float((0.5 * torch.tanh(self.model.prototype_scale)).detach())
            _save_json_atomic(self.epoch_diagnostics,
                              self.log_dir / f"reference_epoch_{epoch:03d}.json")
            print(f"轮次 {epoch:03d} | 验证宏F1 {val_metrics['macro_f1']:.4f} | 用时 {row['epoch_seconds']:.1f} 秒", flush=True)
            eligible_for_best = not self.model.reference or bool(self.model.prototype_ready)
            if eligible_for_best:
                if val_metrics["macro_f1"] > best_score:
                    best_score, best_epoch, stale_epochs = val_metrics["macro_f1"], epoch, 0
                    best_state = {name: value.detach().cpu().clone()
                                  for name, value in self.model.state_dict().items()}
                    if self.args.save_weights:
                        _save_checkpoint_atomic({"model": best_state, "epoch": epoch,
                                                 "prototype_sources": self.prototype_sources,
                                                 "prototype_fallback": self.prototype_fallback}, checkpoint)
                else:
                    stale_epochs += 1
            if self.model.reference and epoch == self.args.warmup_epochs:
                self._select_reservoir(train_loader.dataset)
                self._refresh_prototypes(initial=True)
            if epoch < self.args.warmup_epochs and self.model.reference:
                self.model.prototype_ready.fill_(False)
            _save_checkpoint_atomic({
                "epoch": epoch, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(), "history": history,
                "best_score": best_score, "best_epoch": best_epoch, "stale_epochs": stale_epochs,
                "best_state": best_state, "prototype_sources": self.prototype_sources,
                "prototype_fallback": self.prototype_fallback, "rng": self._rng_state(train_loader),
                "prototype_diagnostics": self.prototype_diagnostics,
                "protocol": protocol,
            }, last_state_path)
            if stale_epochs >= self.args.patience:
                break
        self.model.load_state_dict(best_state)
        return history, best_epoch

    def _run_epoch(self, loader, training):
        
        self.model.train(training)
        losses, labels_all, ids_all, probabilities_all = [], [], [], []
        optimizer_steps, skipped_steps, overflow_count = 0, 0, 0
        diagnostic_values = {key: [] for key in ("prototype_scale",)}
        prototype_active = False
        augmented_batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, (inputs, mask, labels, sample_ids) in enumerate(loader):
                inputs, mask, labels = inputs.to(self.device), mask.to(self.device), labels.to(self.device)
                self.current_labels, self.current_sample_ids = labels, list(sample_ids)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    output = self._forward(inputs, mask, training)
                    loss = self._loss(output, labels)
                for key in diagnostic_values:
                    if key in output:
                        diagnostic_values[key].append(float(output[key].detach().float().mean()))
                if "prototype_active" in output:
                    prototype_active = prototype_active or bool(output["prototype_active"].any())
                augmented_batches += int("augmented" in output)
                if not torch.isfinite(loss) or not torch.isfinite(output["logits"]).all():
                    self._numerical_failure("NEST-MIL非有限损失或预测", batch_index, sample_ids, inputs, output)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grads = [torch.isfinite(parameter.grad).all()
                             for parameter in self.model.parameters() if parameter.grad is not None]
                    finite_gradients = bool(torch.stack(grads).all())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if finite_gradients:
                        self._after_optimizer_step()
                        optimizer_steps += 1
                        overflow_count = 0
                    else:
                        skipped_steps += 1
                        overflow_count += 1
                        if overflow_count >= 8 or not self.scaler.is_enabled():
                            self._numerical_failure("NEST-MIL连续八批梯度非有限", batch_index, sample_ids, inputs, output)
                losses.append(float(loss.detach()))
                labels_all.extend(labels.detach().cpu().tolist())
                batch_probabilities = output["logits"].float().softmax(-1)
                batch_probabilities = batch_probabilities / batch_probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
                probabilities_all.extend(batch_probabilities.detach().cpu().tolist())
                ids_all.extend(sample_ids)
        metrics = classification_metrics(labels_all, probabilities_all,
                                         self.config["num_classes"], self.config["ordinal"])
        metrics["loss"] = float(np.mean(losses))
        metrics["reference_diagnostics"] = {
            key: float(np.mean(values)) for key, values in diagnostic_values.items() if values
        }
        metrics["reference_diagnostics"].update(
            prototype_active=prototype_active, augmented_batches=augmented_batches,
        )
        if training:
            metrics.update(optimizer_steps=optimizer_steps, amp_skipped_steps=skipped_steps,
                           amp_scale=self.scaler.get_scale())
        return metrics, labels_all, np.asarray(probabilities_all).argmax(1).tolist(), ids_all, probabilities_all

    def test(self, loader):
        
        start = perf_counter()
        metrics, labels, predictions, ids, probabilities = self._run_epoch(loader, False)
        result = {'metrics': metrics, 'test_seconds': perf_counter() - start,
                  'predictions': [{'id': name, 'label': label, 'prediction': prediction,
                                   'probabilities': probability}
                                  for name, label, prediction, probability in
                                  zip(ids, labels, predictions, probabilities)]}
        save_json(result, self.log_dir / 'test_result.json')
        return result
