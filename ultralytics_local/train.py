import itertools

import torch
from torch import nn

from ultralytics import YOLO
from argparse import ArgumentParser

from ultralytics.nn.tasks import OBBModel

args = ArgumentParser()

args.add_argument('--resume', type=str, default='')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--model', type=str, default='yolo11-obb-withTransform.yaml')
args.add_argument('--scale', type=str, default='n')
args.add_argument('--imgsz', type=int, default=640)
args.add_argument('--multi_scale', action='store_true', default=False)
args.add_argument('--no_val', action='store_true', default=False)
args.add_argument('--pretrained', type=str, default='')

args = args.parse_args()

args.model = str.replace(args.model, 'yolo11', 'yolo11{0}', 1)
args.model = str.replace(args.model, 'yolo12', 'yolo12{0}', 1)

if len(args.resume) == 0:
    model = YOLO(str.format(args.model, args.scale), task='obb')
    oms: nn.Sequential = model.model.model
    if len(args.pretrained) > 0:
        pretrained: OBBModel = torch.load(args.pretrained)
        for (m1, m2) in itertools.zip_longest(oms, pretrained['model'].model):
            if m1.__class__ == m2.__class__:
                m1.load_state_dict(m2.state_dict())

    results = model.train(data='DOTAv1.5.yaml', epochs=args.epoch, imgsz=args.imgsz, batch=0.6,
                          multi_scale=args.multi_scale, val=not args.no_val)
else:
    model = YOLO(args.resume)

    results = model.train(resume=True, epochs=args.epoch, imgsz=args.imgsz, batch=0.6, multi_scale=args.multi_scale,
                          val=not args.no_val)

if model.trainer.best:
    model.best = model.trainer.best
else:
    model.best = model.trainer.last
