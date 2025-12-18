from dataclasses import dataclass
import torch

@dataclass
class Config:

    ref_line_ratio: float = 0.5

    use_ocr: bool = True
    preprocess_profile: str = "clahe_bilateral_sharp"
    min_plate_len: int = 5
    ocr_min_conf: float = 0.40

    video_in: str = "data/input.mp4"
    video_out: str = "outputs/final_overlay.mp4"


    yolo_vehicle: str = "yolov8n.pt"

    yolo_plate_weights: str = "runs_plate/plate_yolo/weights/best.pt"


    coco_vehicle_class_ids = {2, 3, 5, 7}

    tracker: str = "bytetrack.yaml"
    conf_vehicle: float = 0.35
    iou_vehicle: float = 0.45
    conf_plate: float = 0.30
    iou_plate: float = 0.45

    line_in_y: int = 260
    line_out_y: int = 420
    line_x1: int = 50
    line_x2: int = 1200

    ocr_method: str = "paddle"
    use_preprocess: bool = True
    preprocess_profile: str = "clahe_bilateral_sharp"


    max_frames: int = -1


    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    half: bool = torch.cuda.is_available()

    csv_out: str = "outputs/results.csv"

    log_every: int = 30

