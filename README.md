# bird-detector
## Usage:
`python main.py <args>`

### Argument list:
- `--mode <mode>` two modes:
    - detect - detect birds in the video
    - train - train model
- `--ckpt <path>` - path to trained model
- `--model <model>` - model to be used; if in detect mode ckpt is provided this argument is ignored; possible values:
    - ssd - [ssd300 vgg16](https://pytorch.org/vision/stable/models.html#torchvision.models.detection.ssd300_vgg16)
    - fres - [fasterrcnn resnet50 fpn](https://pytorch.org/vision/stable/models.html#torchvision.models.detection.fasterrcnn_resnet50_fpn)
    - fmob - [fasterrcnn mobilenet_v3_large fpn](https://pytorch.org/vision/stable/models.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn)
- `--video_path <path>` - path to video; must be provided in detect mode
- `--det_score <score>` - value from 0 to 1; score threshold during detection
- `--train_coco <path>` - path to [train coco dataset](http://images.cocodataset.org/zips/train2017.zip); must be provided during training
- `--val_coco <path>` - path to [val coco dataset](http://images.cocodataset.org/zips/val2017.zip); must be provided during training
- `--ann_coco <path>` - path to [annotation folder](http://images.cocodataset.org/annotations/annotations_trainval2017.zip); must be provided during training
- `--epochs <epochs>` - max epochs during training
- `--batch <batch size` - batch size during training
- `--workers <num of workers` - number of workers during training
- `--force_cpu` - run model on cpu; if this flag is not set model will use gpu if one is available
