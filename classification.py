import argparse

import torch
from numpy import require

import experiment

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument(
    '--experiment', choices=['backbone', 'synthesizer', 'multi'], default='backbone')
parser.add_argument('--data-dir', required=True,
                    help='Directory that contains all labels and images')
parser.add_argument('--model-dir', required=False,
                    help='Directory that contains all models')
parser.add_argument('--output-dir', required=False,
                    help='Directory that contains outputs')
parser.add_argument('--focal', action='store_true', default=False)
parser.add_argument(
    '--alpha',
    required=False,
    help='Alpha for focal loss',
    default=0.25)
parser.add_argument(
    '--gamma',
    required=False,
    help='Gamma for focal loss',
    default=2.0)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

device = torch.device('cuda' if args.device == 'cuda' else 'cpu')
if args.experiment == 'backbone':
    expr = experiment.Backbone(device)
elif args.experiment == 'synthesizer':
    expr = experiment.Synthesizer(device)
elif args.experiment == 'multi':
    expr = experiment.Multiclass(device)

expr.prepare_data(args.data_dir)

if args.mode == 'train':
    arg_dict = {}
    if args.focal:
        arg_dict['focal'] = True
        arg_dict['alpha'] = float(args.alpha)
        arg_dict['gamma'] = float(args.gamma)
    expr.train(batch_size=args.batch_size, num_epochs=args.epochs,
               model_dir=args.model_dir, **arg_dict)
elif args.mode == 'test':
    expr.predict(args.model_dir, args.output_dir)
