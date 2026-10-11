import math

import torch
from torch import nn
from torch.nn import functional as F


class RefinementStage(nn.Module):
    

    def __init__(self, hidden_dim, dropout):
        
        super().__init__()
        self.context = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features, mask):
        
        valid = mask.unsqueeze(-1)
        mean = (features * valid).sum(1) / valid.sum(1).clamp_min(1)
        peak = features.masked_fill(~valid, -1e4).max(1).values
        context = self.context(torch.cat([mean, peak], dim=-1)).unsqueeze(1).expand_as(features)
        return self.norm(features + self.update(torch.cat([features, context], dim=-1))) * valid


class NESTMIL(nn.Module):
    

    def __init__(self, num_classes=4, feature_dim=1024, hidden_dim=256, num_stages=3,
                 dropout=0.3, presence_topk=8, presence_temperature=0.25, reference=True,
                 morphology_fp32=True):
        
        super().__init__()
        self.num_classes = num_classes
        self.reference = reference
        self.morphology_fp32 = morphology_fp32
        self.presence_topk = presence_topk
        self.presence_temperature = presence_temperature
        self.projector = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.stages = nn.ModuleList([RefinementStage(hidden_dim, dropout) for _ in range(num_stages)])
        self.instance_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_stages)])
        self.history_weights = nn.Parameter(torch.zeros(num_stages))
        self.question_bias = nn.Parameter(torch.tensor(1.0))
        self.question_alpha_raw = nn.Parameter(torch.tensor(2.0))
        
        self.extent_head = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.threshold_base = nn.Parameter(torch.tensor(-0.5))
        self.threshold_delta = nn.Parameter(torch.tensor(0.5))
        self.morphology_head = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.GELU(),
                                              nn.Dropout(dropout), nn.Linear(128, 3))
        self.arbitration_head = nn.Linear(7, 1)
        self.register_buffer('prototype_bank', torch.zeros(num_classes, 8, hidden_dim))
        self.register_buffer('prototype_ready', torch.tensor(False))
        self.prototype_scale = nn.Parameter(torch.zeros(())) if reference else None
        if num_classes != 4:
            self.morphology_head[-1] = nn.Linear(128, num_classes - 1)
        self.prototype_residual_enabled = True

    def set_prototypes(self, prototypes):
        
        self.prototype_bank.copy_(F.normalize(prototypes.to(self.prototype_bank), dim=-1))
        self.prototype_ready.fill_(True)

    def _reliability(self, stage_logits):
        
        changes = stage_logits.sigmoid()[:, 1:] - stage_logits.sigmoid()[:, :-1]
        values = F.relu(-changes[:, 1:] * changes[:, :-1]).mean(1)
        root = values.float().clamp_min(1e-12).sqrt()
        oscillation = torch.where(values > 0, root, torch.zeros_like(root)).to(values.dtype)
        reliability = torch.sigmoid(self.question_bias - F.softplus(self.question_alpha_raw) * oscillation)
        return reliability, oscillation

    def _extent_summary(self, tumor, mask):
        
        top = tumor.topk(min(128, tumor.size(1)), dim=1).values
        values = [top[:, :width].mean(1) for width in (1, 8, 32, 128)]
        count = mask.sum(1).clamp_min(1)
        return torch.stack(values + [tumor.sum(1) / count, torch.log1p(tumor.sum(1)), torch.log1p(count)], dim=1)

    def forward(self, inputs, mask):
        
        features = self.projector(inputs)
        stage_logits = []
        for stage, head in zip(self.stages, self.instance_heads):
            features = stage(features, mask)
            stage_logits.append(head(features).squeeze(-1))
        stage_logits = torch.stack(stage_logits, dim=1)
        reliability, oscillation = self._reliability(stage_logits)
        instance_logits = (stage_logits * self.history_weights.softmax(0)[None, :, None]).sum(1)
        valid_logits = (instance_logits + reliability.clamp_min(1e-6).log()).masked_fill(~mask, -1e4)
        count = min(self.presence_topk, valid_logits.size(1))
        presence_logit = self.presence_temperature * (
            torch.logsumexp(valid_logits.topk(count, dim=1).values / self.presence_temperature, dim=1)
            - math.log(count)
        )
        tumor = instance_logits.sigmoid() * reliability * mask
        summary = self._extent_summary(tumor, mask)
        burden = self.extent_head(summary).squeeze(-1)
        ordinal_logits = torch.stack([burden - self.threshold_base,
                                      burden - self.threshold_base - F.softplus(self.threshold_delta)], dim=1)
        valid = mask.unsqueeze(-1)
        global_shape = (features * valid).sum(1) / valid.sum(1).clamp_min(1)
        weighted_shape = (features * tumor.unsqueeze(-1)).sum(1) / tumor.sum(1).unsqueeze(1).clamp_min(1e-6)
        morphology_logits = self.morphology_head(torch.cat([global_shape, weighted_shape], dim=1))
        if self.morphology_fp32:
            morphology_logits = morphology_logits.float()
        morphology = morphology_logits.softmax(1)
        evidence = features.new_zeros((*features.shape[:2], self.num_classes - 1), dtype=torch.float32)
        normal_similarity = evidence[:, :, :1].clone()
        positive_similarity = evidence.clone()
        residual = morphology.new_zeros(morphology.shape)
        if self.reference and bool(self.prototype_ready):
            normalized = F.normalize(features.float(), dim=-1)
            bank = F.normalize(self.prototype_bank.float(), dim=-1)
            similarities = torch.einsum('bnd,ckd->bnck', normalized, bank)
            positive_similarity = similarities[:, :, 1:, :].amax(-1)
            normal_similarity = similarities[:, :, 0, :].amax(-1, keepdim=True)
            evidence = F.relu(positive_similarity - normal_similarity - 0.05)
            attention_logits = evidence / 0.1 + tumor.detach().clamp_min(1e-6).log().unsqueeze(-1)
            attention = attention_logits.masked_fill(~mask.unsqueeze(-1), -1e4).softmax(1)
            residual = (attention * evidence).sum(1) * (0.5 * torch.tanh(self.prototype_scale))
        applied_residual = residual if self.prototype_residual_enabled else torch.zeros_like(residual)
        grade = (morphology.clamp_min(1e-8).log() + applied_residual).softmax(1) if self.reference else morphology
        presence = presence_logit.sigmoid()
        probabilities = torch.cat([1 - presence[:, None], presence[:, None] * grade], dim=1)
        output = {'logits': probabilities.clamp_min(1e-8).log(), 'presence_logit': presence_logit,
                  'ordinal_logits': ordinal_logits, 'instance_logits': instance_logits,
                  'instance_probability': instance_logits.sigmoid(), 'reliability': reliability,
                  'oscillation': oscillation, 'extent_summary': summary, 'mask': mask,
                  'global_shape': global_shape, 'weighted_shape': weighted_shape,
                  'prototype_features': features, 'prototype_evidence': applied_residual,
                  'prototype_active': self.prototype_ready.expand(inputs.size(0)),
                  'morphology_logits': morphology_logits, 'conditional_probability': grade,
                  'unapplied_residual': residual, 'tile_evidence': evidence,
                  'normal_similarity': normal_similarity, 'positive_similarity': positive_similarity}
        if self.reference:
            output['prototype_scale'] = (0.5 * torch.tanh(self.prototype_scale)).expand(inputs.size(0))
        return output
