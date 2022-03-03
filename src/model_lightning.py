from pytorch_lightning import LightningModule
from torch.optim import RAdam
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ObjectDetector(LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.save_hyperparameters()
        if model_name == "ssd":
            model = ssd300_vgg16(num_classes=2)
        elif model_name == "fres":
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        elif model_name == "fmob":
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        self.model = model

    def forward(self, x):
        preds = self.model(x)
        return preds

    def configure_optimizers(self):
        return RAdam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss_dict.values())
        loss_dict = {key: item.detach() for key, item in loss_dict.items()}
        loss_dict["loss"] = loss
        # self.log_dict(loss_dict)
        return loss_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred, y

    def validation_epoch_end(self, validation_step_outputs):
        preds, y = zip(*validation_step_outputs)
        preds_2, y_2 = [], []
        for pred in preds:
            preds_2.extend(pred)
        for target in y:
            y_2.extend(target)
        preds = preds_2
        y = y_2
        metric = MeanAveragePrecision()
        metric.update(preds, y)
        self.log_dict({f"val/{key}": item for key, item in metric.compute().items()})
