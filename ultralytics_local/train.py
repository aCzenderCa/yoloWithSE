import itertools

import torch
from torch import nn

from ultralytics import YOLO
from argparse import ArgumentParser

from ultralytics.nn.tasks import OBBModel

args = ArgumentParser()

args.add_argument('--resume', type=str, default='')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--model', type=str, default='yolo11-obb_vitE.yaml')
args.add_argument('--scale', type=str, default='n')
args.add_argument('--imgsz', type=int, default=1024)
args.add_argument('--multi_scale', action='store_true', default=False)
args.add_argument('--no_val', action='store_true', default=False)
args.add_argument('--pretrained', type=str, default='')
args.add_argument('--batch', type=float, default=0.7)
args.add_argument('--optimizer', type=str, default='auto')
args.add_argument('--freeze', type=str, default='')  # 2|4|6|8|9|10|13|16|19|22|23   0|1|2|3|4|5|6
args.add_argument('--lr0', type=float, default=0.1)
args.add_argument('--auto_augment', type=str, default='autoaugment')
args.add_argument('--data', type=str, default='DOTAv1_split.yaml')

args = args.parse_args()

args.model = str.replace(args.model, 'yolo11', 'yolo11{0}', 1)
args.model = str.replace(args.model, 'yolo12', 'yolo12{0}', 1)

train_args = {}
if args.freeze != '':
    train_args['freeze'] = list(map(int, str.split(args.freeze, '|')))
else:
    train_args['freeze'] = []
train_args['epochs'] = args.epoch
train_args['imgsz'] = args.imgsz
train_args['batch'] = args.batch if args.batch < 1 else int(args.batch)
train_args['multi_scale'] = args.multi_scale
train_args['val'] = not args.no_val
train_args['optimizer'] = args.optimizer
train_args['lr0'] = args.lr0
train_args['plots'] = True
train_args['auto_augment'] = args.auto_augment

torch.use_deterministic_algorithms(True, warn_only=False)

if len(args.resume) == 0:
    model = YOLO(str.format(args.model, args.scale), task='obb')
    oms: nn.Sequential = model.model.model
    if len(args.pretrained) > 0:
        pretrained: OBBModel = torch.load(args.pretrained)['model']
        for (m1, m2) in itertools.zip_longest(oms, pretrained.model):
            if m1.__class__ == m2.__class__:
                try:
                    m1.load_state_dict(m2.state_dict(), strict=False, assign=True)
                except Exception as e:
                    print(e)

    results = model.train(data=args.data, **train_args)
else:
    model = YOLO(args.resume)

    results = model.train(resume=True, **train_args)

if model.trainer.best:
    model.best = model.trainer.best
else:
    model.best = model.trainer.last

# python train.py --model last.pt --imgsz 1024 --no_val --optimizer AdamW --lr0 0.001 --multi_scale --batch 0.5 --data DOTAv1_split.yaml --epoch 150
# python train.py --pretrained yolo11n-obb.pt --imgsz 1024 --no_val --optimizer AdamW --lr0 0.002 --model yolo11-obb-withTransform_5.yaml --multi_scale --batch 0.5 --data DOTAv1_split.yaml --freeze "0|1|2|3|4|5" --epoch 50
# python train.py --imgsz 1024 --no_val --optimizer AdamW --lr0 0.002 --model yolo11-obb.yaml --multi_scale --batch 0.5 --data DOTAv1_split.yaml --epoch 50
