from ultralytics_local.ultralytics import YOLO
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument('--resume', type=str, default='')
args.add_argument('--epoch', type=int, default=2)
args.add_argument('--model', type=str, default='yolo11-obb-withTransform.yaml')
args.add_argument('--scale', type=str, default='n')

args = args.parse_args()

args.model = str.replace(args.model, 'yolo11', 'yolo11{0}', 1)
args.model = str.replace(args.model, 'yolo12', 'yolo12{0}', 1)

if len(args.resume) == 0:
    model = YOLO(str.format(args.model, args.scale), task='obb')

    resutls = model.train(data='DOTAv1.yaml', epochs=args.epoch)
    print(resutls)
else:
    model = YOLO(args.resume)

    resutls = model.train(resume=True, epochs=args.epoch)
    print(resutls)
