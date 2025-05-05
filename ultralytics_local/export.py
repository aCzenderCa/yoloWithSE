from ultralytics import YOLO
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument('--model', type=str, default='')
args.add_argument('--fmt', type=str, default='onnx')

args = args.parse_args()

if len(args.model) > 0:
    model = YOLO(args.model)
    model.export(format=args.fmt, dynamic=True)
