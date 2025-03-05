from ultralytics_local.ultralytics import YOLO
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument('--resume', type=str, default='')
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--model', type=str, default='yolo11-obb-withTransform.yaml')
args.add_argument('--scale', type=str, default='n')
args.add_argument('--imgsz', type=int, default=640)
args.add_argument('--multi_scale', type=bool, default=False)
args.add_argument('--no_val', type=bool, action='store_true', default=False)

args = args.parse_args()

args.model = str.replace(args.model, 'yolo11', 'yolo11{0}', 1)
args.model = str.replace(args.model, 'yolo12', 'yolo12{0}', 1)

if len(args.resume) == 0:
    model = YOLO(str.format(args.model, args.scale), task='obb')

    results = model.train(data='DOTAv1.yaml', epochs=args.epoch, imgsz=args.imgsz, batch=0.6,
                          multi_scale=args.multi_scale, val=not args.no_val)
else:
    model = YOLO(args.resume)

    results = model.train(resume=True, epochs=args.epoch, imgsz=args.imgsz, batch=0.6, multi_scale=args.multi_scale,
                          val=not args.no_val)

if model.trainer.best:
    model.best = model.trainer.best
else:
    model.best = model.trainer.last
