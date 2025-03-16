from argparse import ArgumentParser

from ultralytics import YOLO

args = ArgumentParser()

args.add_argument('--model', type=str, default='best.pt')
args.add_argument('--imgsz', type=int, default=1024)
args.add_argument('--dataset', type=str, default='DOTAv1.5.yaml')

args = args.parse_args()

model = YOLO(args.model, task='obb')
model.val(data=args.dataset)
