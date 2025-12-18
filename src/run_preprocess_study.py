import cv2
import pandas as pd
from tqdm import tqdm

from config import Config
from ultralytics import YOLO

from utils_preprocess import preprocess_plate
from utils_metrics import cer, wer, exact_match, normalize_plate

from ocr_paddle import PaddlePlateOCR
from ocr_hunyuan import HunyuanPlateOCR


PREPROCESS_PROFILES = [
    "none",
    "clahe",
    "clahe_bilateral",
    "clahe_bilateral_sharp",
    "gaussian",
    "sharp_only",
    "gray_contrast",
]

OCR_METHODS = ["paddle", "hunyuan"]


def main():
    cfg = Config()
    gt = pd.read_csv("data/gt_plates.csv")
    gt_map = dict(zip(gt["frame_idx"].astype(int), gt["plate_text"].astype(str)))

    vehicle_model = YOLO(cfg.yolo_vehicle)
    plate_model = YOLO(cfg.yolo_plate)

  
    ocr_models = {
        "paddle": PaddlePlateOCR(),
        "hunyuan": HunyuanPlateOCR()
    }

    cap = cv2.VideoCapture(cfg.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video_in}")

    rows = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    for ocr_name in OCR_METHODS:
        ocr = ocr_models[ocr_name]

        for profile in PREPROCESS_PROFILES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            sum_cer, sum_wer, sum_em = 0.0, 0.0, 0
            n = 0

            for frame_idx in tqdm(range(1, total_frames + 1), desc=f"{ocr_name} | {profile}"):
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx not in gt_map:
                    continue  
                gt_text = gt_map[frame_idx]

                
                det = vehicle_model.predict(frame, conf=cfg.conf_vehicle, iou=cfg.iou_vehicle, verbose=False)[0]
                if det.boxes is None or len(det.boxes) == 0:
                    pred_text = ""
                else:

                    vboxes = det.boxes.xyxy.cpu().numpy()
                    vcls = det.boxes.cls.cpu().numpy().astype(int)
                    keep = [i for i in range(len(vcls)) if vcls[i] in cfg.coco_vehicle_class_ids]
                    if not keep:
                        pred_text = ""
                    else:
                        vboxes = vboxes[keep]
                        areas = (vboxes[:,2]-vboxes[:,0])*(vboxes[:,3]-vboxes[:,1])
                        k = int(areas.argmax())
                        x1,y1,x2,y2 = vboxes[k].astype(int)
                        roi = frame[y1:y2, x1:x2]

                        pred_text = ""
                        if roi.size != 0:
                            pres = plate_model.predict(roi, conf=cfg.conf_plate, iou=cfg.iou_plate, verbose=False)[0]
                            if pres.boxes is not None and len(pres.boxes) > 0:
                                pboxes = pres.boxes.xyxy.cpu().numpy()
                                areas_p = (pboxes[:,2]-pboxes[:,0])*(pboxes[:,3]-pboxes[:,1])
                                kp = int(areas_p.argmax())
                                px1,py1,px2,py2 = pboxes[kp].astype(int)
                                plate_crop = roi[py1:py2, px1:px2]
                                if plate_crop.size != 0:
                                    img_for_ocr = preprocess_plate(plate_crop, profile)
                                    pred_text, _ = ocr.recognize(img_for_ocr)
                                    pred_text = normalize_plate(pred_text)

                sum_cer += cer(pred_text, gt_text)
                sum_wer += wer(pred_text, gt_text)
                sum_em += exact_match(pred_text, gt_text)
                n += 1

            if n == 0:
                avg_cer, avg_wer, acc = 1.0, 1.0, 0.0
            else:
                avg_cer = sum_cer / n
                avg_wer = sum_wer / n
                acc = sum_em / n

            rows.append({
                "ocr": ocr_name,
                "preprocess": profile,
                "samples": n,
                "CER": avg_cer,
                "WER": avg_wer,
                "ExactMatchAcc": acc
            })

    cap.release()

    df = pd.DataFrame(rows).sort_values(["CER", "WER"])
    out_path = "outputs/tables/ocr_comparison.csv"
    df.to_csv(out_path, index=False)
    print(df.head(15))
    print(f"[DONE] table saved: {out_path}")


if __name__ == "__main__":
    main()
