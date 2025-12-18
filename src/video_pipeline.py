import os
import csv
import time
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

from utils_text import normalize_plate

VIDEO_IN = "data/input.mp4"
VIDEO_OUT = "outputs/final_overlay.mp4"
CSV_OUT = "outputs/results.csv"

VEHICLE_MODEL = "yolov8n.pt"        
PLATE_MODEL = "runs_plate/plate_yolo/weights/best.pt" 

CONF_VEHICLE = 0.35
CONF_PLATE = 0.3

COCO_VEHICLES = {2, 3, 5, 7}

REF_LINE_RATIO = 0.55        
LOG_EVERY = 30                

def clamp(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2


def main():
    os.makedirs("outputs", exist_ok=True)

    print("[INFO] Loading models...")
    vehicle_model = YOLO(VEHICLE_MODEL)
    plate_model = YOLO(PLATE_MODEL)

    print("[INFO] Initializing EasyOCR...")
    ocr = easyocr.Reader(
        ["en"],
        gpu=False,
        recog_network="latin_g2"
    )

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    ref_y = int(h * REF_LINE_RATIO)

    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    csv_f = open(CSV_OUT, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(["frame", "track_id", "direction", "plate", "confidence"])

    last_y = {}
    direction = {}
    best_plate = {}

    cnt_towards = 0
    cnt_away = 0

    frame_idx = 0
    t0 = time.time()

    print("[INFO] Processing video...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        if frame_idx % LOG_EVERY == 0:
            print(f"[INFO] frame={frame_idx} towards={cnt_towards} away={cnt_away}")

        res = vehicle_model.track(
            frame,
            persist=True,
            conf=CONF_VEHICLE,
            verbose=False
        )

        cv2.line(frame, (0, ref_y), (w, ref_y), (0, 255, 255), 2)

        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                if b.id is None:
                    continue

                cls = int(b.cls[0].item())
                if cls not in COCO_VEHICLES:
                    continue

                tid = int(b.id[0].item())
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                x1, y1, x2, y2 = clamp(x1, y1, x2, y2, w, h)

                cy = (y1 + y2) // 2

         
                if tid in last_y:
                    dy = cy - last_y[tid]
                    if tid not in direction:
                        direction[tid] = "TOWARDS" if dy > 0 else "AWAY"

                    crossed = (last_y[tid] < ref_y <= cy) or (last_y[tid] > ref_y >= cy)
                    if crossed:
                        if direction[tid] == "TOWARDS":
                            cnt_towards += 1
                        else:
                            cnt_away += 1

                last_y[tid] = cy

           
                car = frame[y1:y2, x1:x2]
                if car.size > 0:
                    pres = plate_model.predict(car, conf=CONF_PLATE, verbose=False)

                    if pres and pres[0].boxes is not None and len(pres[0].boxes) > 0:
                        p = pres[0].boxes.xyxy[0].cpu().numpy()
                        px1, py1, px2, py2 = clamp(
                            p[0], p[1], p[2], p[3], car.shape[1], car.shape[0]
                        )

                        plate = car[py1:py2, px1:px2]

                        if plate.size > 0:
                            ocr_res = ocr.readtext(plate, detail=1, paragraph=False)
                            for _, txt, conf in ocr_res:
                                norm = normalize_plate(txt)
                                if norm:
                                    old = best_plate.get(tid, ("", 0))
                                    if conf > old[1]:
                                        best_plate[tid] = (norm, conf)
                                        writer.writerow(
                                            [frame_idx, tid, direction.get(tid), norm, round(conf, 3)]
                                        )
                                        print(f"[OCR] {tid} → {norm} ({conf:.2f})")

              
                        cv2.rectangle(
                            frame,
                            (x1 + px1, y1 + py1),
                            (x1 + px2, y1 + py2),
                            (0, 255, 0),
                            2
                        )

          
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                label = f"ID:{tid} {direction.get(tid,'')}"
                if tid in best_plate:
                    label += f" {best_plate[tid][0]}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        cv2.putText(
            frame,
            f"Towards:{cnt_towards} Away:{cnt_away}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        out.write(frame)

    cap.release()
    out.release()
    csv_f.close()

    print("[DONE] Video:", VIDEO_OUT)
    print("[DONE] CSV:", CSV_OUT)
    print(f"[TIME] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
