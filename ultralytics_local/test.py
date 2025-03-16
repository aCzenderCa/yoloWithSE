from argparse import ArgumentParser
from ultralytics.utils.benchmarks import benchmark

import torch
from ultralytics import YOLO

args = ArgumentParser()

args.add_argument('--model', type=str, default='best.pt')
args.add_argument('--imgsz', type=int, default=1024)
args.add_argument('--dataset', type=str, default='DOTAv1.5.yaml')
args.add_argument('--only_model_structure', action='store_true', default=False)

args = args.parse_args()

args.model = str.split(args.model, '|')

for model in args.model:
    print(model, end="")
    if args.only_model_structure:
        model = YOLO(model, task='obb')
        model.predict(torch.zeros(3, 3, args.imgsz, args.imgsz))
    else:
        benchmark(model=model, data=args.dataset, imgsz=args.imgsz, half=True)
        metrics = model.val(data=args.dataset)
        print(f"mAP50:{metrics.box.map50}")

    print()
