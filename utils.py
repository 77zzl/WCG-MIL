import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_hash(path):
    
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(content, path):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(content, path):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(content, temporary)
    os.replace(temporary, path)


def load_model_checkpoint(model, path):
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return checkpoint
