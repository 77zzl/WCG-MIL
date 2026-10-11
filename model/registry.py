from functools import partial
from .nestmil import NESTMIL

MODEL_REGISTRY = {'nest-mil': NESTMIL, 'nest-mil-core': partial(NESTMIL, reference=False)}


def create_model(name, config, args):
    
    return MODEL_REGISTRY[name](num_classes=config['num_classes'], hidden_dim=args.hidden_dim,
                               dropout=args.dropout, presence_topk=args.presence_topk,
                               presence_temperature=args.presence_temperature,
                               morphology_fp32=name == 'nest-mil' or config['num_classes'] == 3)
