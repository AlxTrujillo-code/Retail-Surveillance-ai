import torch
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.dataloaders import LoadImages

# CONFIG
video_path = "input_videos/Shoplifting016_full.mp4"  # Change as needed
weights = "yolov5s.pt"
data_yaml = "data/coco128.yaml"

# Load model
device = select_device('')
model = DetectMultiBackend(weights, device=device, data=data_yaml)
stride, names, _ = model.stride, model.names, model.pt
model.warmup(imgsz=(1, 3, 640, 640))

# Load video
dataset = LoadImages(video_path, img_size=640, stride=stride, auto=True)

# Run detection loop
for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    pred = model(im)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        annotator = Annotator(im0s.copy(), line_width=2, example=str(names))
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
        frame = annotator.result()

    cv2.imshow("YOLOv5 Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
