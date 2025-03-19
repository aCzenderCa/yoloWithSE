from ultralytics.data.split_dota import split_test, split_trainval
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument('--data', type=str, default='')
args.add_argument('--max_count', type=float, default=10000)

args = args.parse_args()

if len(args.data) > 0:
    # split train and val set, with labels.
    split_trainval(
        data_root=args.data,
        save_dir=args.data + "_split",
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
        max_count=args.max_count,
    )
