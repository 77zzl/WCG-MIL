import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from args import get_arguments
from data import build_loaders
from engine import NESTTrainer
from model.registry import create_model
from utils import file_hash, load_model_checkpoint, save_json, set_seed


def measure_efficiency(model, loader, device):
    
    inputs, mask, _, _ = next(iter(loader))
    inputs, mask = inputs.to(device), mask.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(inputs, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for _ in range(10):
            model(inputs, mask)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    result = {'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'batch_forward_ms': (perf_counter() - start) * 100}
    if device.type == 'cuda':
        result['gpu_peak_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
    return result


def main():
    
    args = get_arguments()
    if args.dataset == 'bracs' and args.fold != 1:
        raise ValueError('BRACS使用单个官方划分，重复单位是seed而不是fold。')
    if args.resume_training and not args.run_id:
        raise ValueError('恢复训练必须提供原run-id，避免把旧状态写入新任务。')
    if not args.checkpoint and args.disable_reference_residual:
        raise ValueError('关闭参照残差仅用于冻结评估，不用于论文模型训练。')
    if not args.checkpoint and (args.warmup_epochs < 1 or args.epochs <= args.warmup_epochs) and args.name == 'nest-mil':
        raise ValueError('NEST-MIL需在预热后继续训练并保存有效参照权重。')
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    args.optimizer, args.num_stages, args.loss_mode = 'adamw', 3, 'joint_nll'
    args.feature_name, args.feature_bag_size, args.batch_size, args.class_balance = 'UNI', 0, 1, 'effective'
    args.split_protocol = 'grouped_5fold' if args.dataset == 'camelyon_plus' else 'official_roi'
    args.save_weights = True
    directory = Path(args.output_dir) / args.dataset / args.name / f'fold_{args.fold}' / f'seed_{args.seed}' / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'run_summary.json').exists():
        raise FileExistsError('该run已完成，请使用新run-id或独立评估目录。')
    loaders, config, split = build_loaders(args)
    args.split_hash = file_hash(split)
    sources = sorted(Path(__file__).parent.glob('*.py')) + sorted((Path(__file__).parent / 'model').glob('*.py'))
    import hashlib
    args.source_hash = hashlib.sha256(json.dumps([(p.name, file_hash(p)) for p in sources]).encode()).hexdigest()
    save_json({'arguments': vars(args), 'split_sizes': [len(loader.dataset) for loader in loaders],
               'torch_version': torch.__version__, 'device': str(device)}, directory / 'run_info.json')
    model = create_model(args.name, config, args).to(device)
    trainer = NESTTrainer(model, args, config, device, directory, directory)
    start = perf_counter()
    if args.checkpoint:
        checkpoint = load_model_checkpoint(model, args.checkpoint)
        model.prototype_residual_enabled = not args.disable_reference_residual
        trainer._prepare_training_protocol(loaders[0])
        best_epoch = checkpoint.get('epoch', None)
        fit_seconds = 0.0
        history = []
    else:
        history, best_epoch = trainer.fit(loaders[0], loaders[1])
        fit_seconds = perf_counter() - start
    if args.efficiency:
        save_json(measure_efficiency(model, loaders[2], device), directory / 'efficiency.json')
    result = trainer.test(loaders[2])
    save_json({'best_epoch': best_epoch, 'actual_epochs': len(history), 'max_epochs': args.epochs,
               'fit_seconds': fit_seconds, 'total_seconds': perf_counter() - start,
               'termination': 'evaluation_only' if args.checkpoint else ('early_stop' if len(history) < args.epochs else 'epoch_limit'),
               'metrics': result['metrics']}, directory / 'run_summary.json')
    print(f'完成：{directory}', flush=True)


if __name__ == '__main__':
    main()
