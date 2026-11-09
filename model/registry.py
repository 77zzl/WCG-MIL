from .wcgmil import WCGMIL

MODEL_REGISTRY = {'wcg-mil': WCGMIL}


def create_model(name, config, args):
    
    return MODEL_REGISTRY[name](num_classes=config['num_classes'], hidden_dim=args.hidden_dim,
                               dropout=args.dropout, presence_topk=args.presence_topk,
                               presence_temperature=args.presence_temperature)
