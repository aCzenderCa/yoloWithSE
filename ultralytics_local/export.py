from ultralytics import YOLO
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument('--model', type=str, default='')

args = args.parse_args()

if len(args.model) > 0:
    model = YOLO(args.model)
    model.export(format='onnx')
