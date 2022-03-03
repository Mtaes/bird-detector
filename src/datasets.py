import json
import os

import cv2
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    convert_image_dtype,
    pil_to_tensor,
    to_tensor,
)


class VideoDataset(Dataset):
    def __init__(self, path_to_video: str):
        super().__init__()
        self.cap = cv2.VideoCapture(path_to_video)

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return to_tensor(frame)

    def __del__(self):
        self.cap.release()


class COCOBirdsDataset(Dataset):
    def __init__(self, meta_path: str, images_path: str):
        super().__init__()
        with open(meta_path) as meta:
            coco_meta = json.load(meta)
        images = {}
        for ann in coco_meta["annotations"]:
            if ann["category_id"] == 16:
                bbox = ann["bbox"]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                if ann["image_id"] in images:
                    images[ann["image_id"]].append(bbox)
                else:
                    images[ann["image_id"]] = [bbox]
        tmp_imgs = []
        for key, item in images.items():
            file_name = next(
                x["file_name"] for x in coco_meta["images"] if x["id"] == key
            )
            tmp_imgs.append(
                {
                    "file_name": file_name,
                    "bboxes": item,
                }
            )
        self.images = tmp_imgs
        self.images_path = images_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        img = Image.open(os.path.join(self.images_path, item["file_name"])).convert(
            "RGB"
        )
        img = pil_to_tensor(img)
        img = convert_image_dtype(img)
        targets = {
            "boxes": tensor(item["bboxes"]),
            "labels": tensor([1] * len(item["bboxes"])),
        }
        return img, targets
