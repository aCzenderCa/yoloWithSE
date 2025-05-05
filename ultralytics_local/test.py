from argparse import ArgumentParser
from ultralytics.utils.benchmarks import benchmark

import torch
from ultralytics import YOLO

args = ArgumentParser()

args.add_argument('--model', type=str, default='best.pt')
args.add_argument('--imgsz', type=int, default=1024)
args.add_argument('--dataset', type=str, default='DOTAv1_split.yaml')
args.add_argument('--only_model_structure', action='store_true', default=False)
args.add_argument('--device', type=str, default=None)

args = args.parse_args()

args.model = str.split(args.model, '|')

for model in args.model:
    print(model, end="")
    model = YOLO(model, task='obb')
    if args.only_model_structure:
        model.predict(torch.zeros(3, 3, args.imgsz, args.imgsz))
    else:
        metrics = model.val(data=args.dataset, imgsz=args.imgsz)
        print(f"mAP50:{metrics.box.map50}")

    print()
