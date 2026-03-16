# 1.0-run_pipeline.py
import os
import json
import sys

PROJECT_ROOT = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT"
sys.path.append(PROJECT_ROOT)

from ultralytics import YOLO

from detect_panels import detect_panels_xyxy
from match_panels import greedy_match, assign_row_col
from crop_panels import crop_matched_panels

# ==== 實際路徑 ====
RGB_MODEL_PATH = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/runs/detect/train/yolo11-RGB-module/fold3/weights/best.pt"
IR_MODEL_PATH  = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/runs/detect/train/yolo11-IR-module/fold3/weights/best.pt"

RGB_DIR = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsModule/RGB/AllImage"
IR_DIR  = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsModule/IR/AllImage"

OUT_ROOT = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsPairs/PairsImages"

CONF_THRES = 0.5
ROW_EPS    = 0.03
MAX_DIST   = 0.05  # normalized center distance 閾值


def collect_pairs(rgb_dir: str, ir_dir: str):
    """
    假設 RGB / IR 檔名相同，例如：
        RGB/AllImage/1-1.jpg
        IR/AllImage/1-1.jpg

    以「去掉副檔名的檔名」當 key 來配對。
    """
    rgb_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(rgb_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    }

    ir_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(ir_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    }

    pairs = []
    for stem, rgb_f in rgb_files.items():
        if stem in ir_files:
            ir_f = ir_files[stem]
            pairs.append((stem, rgb_f, ir_f))
        else:
            print(f"[警告] 找不到對應 IR：{rgb_f}")

    # 依檔名排序方便 debug
    return sorted(pairs, key=lambda x: x[0])


def main():
    print(f"[INFO] 載入 RGB 模型：{RGB_MODEL_PATH}")
    rgb_model = YOLO(RGB_MODEL_PATH)

    print(f"[INFO] 載入 IR 模型：{IR_MODEL_PATH}")
    ir_model = YOLO(IR_MODEL_PATH)

    pairs = collect_pairs(RGB_DIR, IR_DIR)
    print(f"[INFO] 發現 {len(pairs)} 組 RGB–IR 圖片")

    os.makedirs(OUT_ROOT, exist_ok=True)

    for stem, rgb_file, ir_file in pairs:
        print(f"[INFO] 處理場景：{stem}")

        rgb_path = os.path.join(RGB_DIR, rgb_file)
        ir_path  = os.path.join(IR_DIR,  ir_file)
        out_dir  = os.path.join(OUT_ROOT, stem)

        # Step 1: YOLO 偵測 panel
        rgb_det = detect_panels_xyxy(rgb_model, rgb_path, conf_thres=CONF_THRES)
        ir_det  = detect_panels_xyxy(ir_model,  ir_path,  conf_thres=CONF_THRES)

        rgb_panels = rgb_det["panels"]
        ir_panels  = ir_det["panels"]

        if not rgb_panels or not ir_panels:
            print(f"[警告] {stem}：RGB 或 IR 沒偵測到 panel，略過裁切")
            continue

        # Step 2: greedy 配對 + row/col 命名
        matched_pairs = greedy_match(rgb_panels, ir_panels, max_dist=MAX_DIST)
        id_map = assign_row_col(rgb_panels, row_eps=ROW_EPS)

        # Step 3: 裁切 panel
        crop_matched_panels(
            rgb_path, ir_path,
            rgb_panels, ir_panels,
            matched_pairs, id_map,
            out_dir,
        )

        # Step 4: 寫入 match.json
        match_info = {
            "meta": {
                "rgb_path": rgb_path,
                "ir_path": ir_path,
                "num_rgb": len(rgb_panels),
                "num_ir": len(ir_panels),
                "num_matched": len(matched_pairs),
                "max_dist": MAX_DIST,
            },
            "pairs": matched_pairs,
            "id_map": id_map,
            "rgb_panels": rgb_panels,
            "ir_panels": ir_panels,
        }

        with open(os.path.join(out_dir, "match.json"), "w", encoding="utf-8") as f:
            json.dump(match_info, f, indent=2, ensure_ascii=False)

    print("[INFO] 全部場景處理完畢。")


if __name__ == "__main__":
    main()
