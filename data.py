import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATASETS = {'camelyon_plus': ['negative', 'itc', 'micro', 'macro'],
            'bracs': ['benign', 'atypical', 'malignant']}


class FeatureBagDataset(Dataset):
    

    def __init__(self, records):
        
        self.records = records

    def __len__(self):
        
        return len(self.records)

    def __getitem__(self, index):
        
        path, label, sample_id = self.records[index]
        features = torch.load(path, map_location='cpu', weights_only=True).float()
        return features, torch.ones(features.size(0), dtype=torch.bool), label, sample_id


def read_split(args):
    
    filename = f'fold_{args.fold}.csv' if args.dataset == 'camelyon_plus' else 'official.csv'
    path = Path(__file__).parent / 'splits' / args.dataset / filename
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = {split: {row['group'] for row in rows if row['split'] == split}
              for split in ('train', 'val', 'test')}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if groups[left] & groups[right]:
            raise ValueError(f'划分清单出现跨集合组：{left}/{right}')
    return rows, path


def build_loaders(args):
    
    rows, path = read_split(args)
    datasets = [FeatureBagDataset([(Path(args.feature_root) / f"{row['id']}.pt", int(row['label']), row['id'])
                                  for row in rows if row['split'] == split])
                for split in ('train', 'val', 'test')]
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(dataset, batch_size=1, shuffle=index == 0, num_workers=args.num_workers,
                          pin_memory=True, generator=generator if index == 0 else None)
               for index, dataset in enumerate(datasets)]
    classes = DATASETS[args.dataset]
    return loaders, {'classes': classes, 'num_classes': len(classes), 'ordinal': args.dataset == 'camelyon_plus'}, path
