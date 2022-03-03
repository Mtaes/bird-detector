import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)

from datasets import COCOBirdsDataset
from model_lightning import ObjectDetector
from utils import collate_fn, detect, get_args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "detect":
        target = 16
        if args.ckpt is not None:
            model = ObjectDetector.load_from_checkpoint(args.ckpt)
            target = 1
        elif args.model == "ssd":
            model = ssd300_vgg16(pretrained=True)
        elif args.model == "fres":
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        elif args.model == "fmob":
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        device = torch.device("cpu") if args.force_cpu else None
        detect(model, str(args.video_path), target, args.det_score, device)
    elif args.mode == "train":
        assert args.epochs > 0
        assert args.batch > 0
        assert args.workers >= 0
        dataset_train = COCOBirdsDataset(
            args.ann_coco / "instances_train2017.json", args.train_coco
        )
        dataset_val = COCOBirdsDataset(
            args.ann_coco / "instances_val2017.json", args.val_coco
        )
        dataloader_train = DataLoader(
            dataset_train,
            collate_fn=collate_fn,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True,
        )
        dataloader_val = DataLoader(
            dataset_val,
            collate_fn=collate_fn,
            batch_size=args.batch,
            num_workers=args.workers,
            pin_memory=True,
        )
        model = ObjectDetector(args.model)
        gpus = 0 if args.force_cpu or not torch.cuda.is_available() else 1
        trainer = Trainer(
            gpus=gpus,
            max_epochs=args.epochs,
            callbacks=[ModelCheckpoint(monitor="val/map", mode="max")],
            log_every_n_steps=1,
        )
        trainer.validate(model, dataloader_val)
        trainer.fit(model, dataloader_train, dataloader_val)
