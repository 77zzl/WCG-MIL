import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import save_json


Image.MAX_IMAGE_PIXELS = None


PROJECT = Path(__file__).resolve().parents[1]
ROOT = None
CACHE = None
LABELS = {"0_N": 0, "1_PB": 0, "2_UDH": 0, "3_FEA": 1,
          "4_ADH": 1, "5_DCIS": 2, "6_IC": 2}


def read_json(path):
    
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value, path):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(value, temporary)
    os.replace(temporary, path)


def audit_rois(root=None, require_images=True):
    
    root = Path(root or ROOT)
    table = pd.read_excel(root / "BRACS.xlsx", sheet_name="WSI_Information")
    lookup = table.set_index("WSI Filename").to_dict("index")
    names = {"train": "Training", "val": "Validation", "test": "Testing"}
    records = []
    damaged = []
    patients = {}
    for split in names:
        patients[split] = set()
        for path in sorted((root / split).glob("*/*.png")):
            wsi = "_".join(path.stem.split("_")[:2])
            info = lookup[wsi]
            if info["Set"] != names[split]:
                raise ValueError(f"ROI与官方集合不一致：{path}")
            patient = str(int(info["Patient Id"]))
            patients[split].add(patient)
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, ValueError) as error:
                width, height = 0, 0
                damaged.append({"path": str(path), "bytes": path.stat().st_size,
                                "split": split, "id": path.stem, "reason": str(error)})
            records.append({"id": path.stem, "split": split, "label": LABELS[path.parent.name],
                            "subtype": path.parent.name, "patient": patient, "wsi": wsi,
                            "image": str(path), "width": width, "height": height})
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if patients[left] & patients[right]:
            raise ValueError(f"BRACS出现跨集合患者：{left}/{right}")
    atomic_json({"damaged": damaged, "images": len(records), "patient_overlap": False},
                PROJECT / "output/bracs_data_audit.json")
    if damaged and require_images:
        raise ValueError(f"BRACS有{len(damaged)}份损坏ROI，不允许静默丢弃。详见output/bracs_data_audit.json；修复原文件后重跑。")
    return records, {split: {"rois": sum(r["split"] == split for r in records),
                            "patients": len(patients[split]),
                            "classes": [sum(r["split"] == split and r["label"] == c for r in records)
                                        for c in range(3)]} for split in names}


def tile_coordinates(width, height):
    
    return [(x, y) for y in range(0, height, 256) for x in range(0, width, 256)]


def get_tile(image, x, y):
    
    tile = Image.new("RGB", (256, 256), "white")
    tile.paste(image.crop((x, y, min(x + 256, image.width), min(y + 256, image.height))), (0, 0))
    return tile


def resolve_uni_checkpoint(requested=None):
    
    path = Path(requested) if requested else Path(os.environ.get('NEST_UNI_CHECKPOINT', 'weights/UNI/pytorch_model.bin'))
    if not path.is_file():
        raise FileNotFoundError('请用--uni-checkpoint指定获授权的官方UNI pytorch_model.bin。')
    return path


def prepare_cache(checkpoint=None, batch_size=16):
    
    records, audit = audit_rois()
    checkpoint = resolve_uni_checkpoint(checkpoint)
    protocol = {"encoder": "MahmoodLab/UNI", "checkpoint_sha256": file_hash(checkpoint),
                "tile": 256, "resize": 224, "padding": "white", "filter": "none",
                "normalization": "ImageNet", "feature_dim": 1024, "precision": "float32"}
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["protocol"] != protocol:
        raise ValueError("已有BRACS缓存的编码器或预处理不同，禁止混用。请先另行归档旧缓存。")
    atomic_json({"protocol": protocol, "checkpoint": str(checkpoint), "audit": audit,
                 "status": "extracting", "records": records}, manifest_path)
    import timm
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16,
                              init_values=1e-5, num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    for index, row in enumerate(records, 1):
        output = CACHE / "features" / f"{row['id']}.pt"
        metadata = CACHE / "tiles" / f"{row['id']}.json"
        if output.exists() and metadata.exists():
            continue
        with Image.open(row["image"]) as raw:
            image = raw.convert("RGB")
        coordinates = tile_coordinates(*image.size)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(coordinates), batch_size):
                values = torch.stack([transform(get_tile(image, x, y))
                                      for x, y in coordinates[start:start + batch_size]]).cuda()
                batches.append(model(values).float().cpu())
        features = torch.cat(batches)
        if not torch.isfinite(features).all() or features.size(1) != 1024:
            raise ValueError(f"UNI特征异常：{row['id']}")
        output.parent.mkdir(exist_ok=True)
        temporary = output.with_suffix(".tmp")
        torch.save(features, temporary)
        os.replace(temporary, output)
        atomic_json({**row, "coordinates": coordinates, "tile_size": 256,
                     "encoder_sha256": protocol["checkpoint_sha256"]}, metadata)
        print(f"BRACS共享缓存 {index}/{len(records)}：{row['id']}，{len(coordinates)}块", flush=True)
    atomic_json({"protocol": protocol, "checkpoint": str(checkpoint), "audit": audit,
                 "status": "completed", "records": records}, manifest_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Frozen UNI features for official BRACS ROIs')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--uni-checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    arguments = parser.parse_args()
    ROOT, CACHE = Path(arguments.data_root), Path(arguments.cache_dir)
    prepare_cache(arguments.uni_checkpoint, arguments.batch_size)
