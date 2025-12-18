import os
import time
import zipfile
import random
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

REPO_ID = "AY000554/Car_plate_detecting_dataset"

def collect_images(root):
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for d, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(d, f))
    return paths

def download_and_extract(dataset_root="datasets/Car_plate_detecting_dataset"):
    os.makedirs(dataset_root, exist_ok=True)
    splits = ["train", "val", "test"]
    raw_dirs = {}

    print("[1] Download+extract dataset...")
    for split in splits:
        zip_path = hf_hub_download(repo_id=REPO_ID, filename=f"{split}.zip", repo_type="dataset")
        split_dir = os.path.join(dataset_root, f"{split}_raw")
        raw_dirs[split] = split_dir
        os.makedirs(split_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(split_dir)
    return raw_dirs

def prepare_yolo_dataset(raw_dirs, out_root="datasets/plates_yolo", train_fraction=0.2):
    print("[2] Prepare YOLO dataset...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    def copy_split(split, frac):
        raw = raw_dirs[split]
        imgs = collect_images(raw)
        if frac < 1.0:
            use = random.sample(imgs, max(1, int(len(imgs) * frac)))
        else:
            use = imgs

        for img in use:
            base = os.path.splitext(os.path.basename(img))[0]
            label = None
            for d, _, files in os.walk(raw):
                if base + ".txt" in files:
                    label = os.path.join(d, base + ".txt")
                    break
            if label:
                shutil.copy2(img, os.path.join(out_root, "images", split, os.path.basename(img)))
                shutil.copy2(label, os.path.join(out_root, "labels", split, base + ".txt"))

        print(f"    {split}: {len(use)} images")

    copy_split("train", train_fraction)
    copy_split("val", 1.0)
    copy_split("test", 1.0)

    abs_path = os.path.abspath(out_root).replace("\\", "/")
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {abs_path}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "nc: 1\n"
            "names: [plate]\n"
        )

    print("[2] YAML:", yaml_path)
    return yaml_path

def train(data_yaml, device="0"):
    print("[3] Train YOLO for plates...")
    model = YOLO("yolov8n.pt")
    t0 = time.time()

    model.train(
        data=data_yaml,
        epochs=3,          # для экзамена лучше 15-30
        imgsz=512,
        batch=16,
        device=device,
        project="runs_plate",
        name="plate_yolo",
        exist_ok=True
    )

    t1 = time.time()
    best = os.path.join("runs_plate", "plate_yolo", "weights", "best.pt")
    print(f"[3] Done in {t1-t0:.1f}s best={best}")
    return best

def main():
    raw = download_and_extract()
    yaml_path = prepare_yolo_dataset(raw, out_root="datasets/plates_yolo", train_fraction=0.2)
    best = train(yaml_path, device="0")

    os.makedirs("weights", exist_ok=True)
    dst = os.path.join("weights", "plate.pt")
    shutil.copy2(best, dst)
    print("[DONE] Saved weights ->", dst)

if __name__ == "__main__":
    main()
