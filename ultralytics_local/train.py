from ultralytics_local.ultralytics import YOLO
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument('--resume', type=str, default='')
args.add_argument('--epoch', type=int, default=2)
args.add_argument('--model', type=str, default='yolo11-obb-withTransform.yaml')

args = args.parse_args()

if len(args.resume) == 0:
    model = YOLO(args.model)

    resutls = model.train(data='DOTAv1.yaml', epochs=args.epoch)
    print(resutls)
else:
    model = YOLO(args.resume)

    resutls = model.train(resume=True, epochs=args.epoch)
    print(resutls)
