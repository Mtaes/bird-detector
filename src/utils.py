from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VideoDataset


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["detect", "train"], required=True)
    parser.add_argument(
        "--model", type=str, choices=["ssd", "fres", "fmob"], default="ssd"
    )
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--video_path", type=Path)
    parser.add_argument("--det_score", type=float, default=0.5)
    parser.add_argument("--train_coco", type=Path)
    parser.add_argument("--val_coco", type=Path)
    parser.add_argument("--ann_coco", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    return args


def draw_bbox(pil_img, text, box, color, font, frame_width):
    xy = (box[:2] + frame_width).tolist()
    img = ImageDraw.Draw(pil_img)
    text_size = img.textsize(text, font=font)
    text_bbox = xy + [xy[0] + text_size[0], xy[1] + text_size[1]]
    if (diff := text_bbox[2] - pil_img.size[0]) > 0:
        xy[0] -= diff
        text_bbox[0] -= diff
        text_bbox[2] -= diff
    color = tuple(int(i * 255) for i in color)
    img.rectangle(box.tolist(), outline=color, width=frame_width)
    img.rectangle(text_bbox, fill=color)
    img.text(xy, text, fill=(255, 255, 255, 255), font=font)


def detect(
    model,
    in_path: str,
    target: int,
    det_score: float,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    vd = VideoDataset(in_path)
    vdl = DataLoader(vd, pin_memory=True)
    model.eval()
    video_res = (
        int(vd.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vd.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = int(vd.cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(
        in_path + "-detections.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, video_res
    )
    cmap = plt.get_cmap("tab20")
    font = ImageFont.truetype("arial.ttf", 40)
    with torch.no_grad():
        model.to(device)
        for frame in tqdm(vdl):
            dets = model(frame.to(device))
            frame = (frame[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame = Image.fromarray(frame)
            dets = dets[0]
            for box, score, label in zip(dets["boxes"], dets["scores"], dets["labels"]):
                if label != target or score.item() < det_score:
                    continue
                box = box.tolist()
                color = cmap(17)[:3]
                draw_bbox(frame, "bird", torch.Tensor(box), color, font, frame_width=7)
            out.write(np.array(frame))
    out.release()
    cv2.destroyAllWindows()


def collate_fn(batch):
    return tuple(zip(*batch))
