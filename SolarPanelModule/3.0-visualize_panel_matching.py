# 功能：
# 1. 讀取每個場景資料夾下的 match.json（由配對程式 1.0-run_pipeline.py 產生）
# 2. 載入整張 RGB / IR 場景圖，左邊放「縮小的 IR」，右邊放「原尺寸 RGB」
# 3. 左邊 IR 圖畫出每個 panel 的 bbox + rX_cY 標籤（使用 match.json 中的 ir_panels）
# 4. 右邊 RGB 圖畫出對應 panel 的 bbox + rX_cY 標籤（使用 match.json 中的 rgb_panels）
# 5. 對每個成功配對的 panel，從 IR panel 中心連到 RGB panel 中心畫線
# 6. 每個場景輸出一張可視化圖：<scene>/vis_rgb_ir_pairs.jpg

import os
import sys
import json
import cv2
import numpy as np

PROJECT_ROOT = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT"
sys.path.append(PROJECT_ROOT)

# 這裡是你 panel 資料夾 root（裡面有 1-1, 1-2, ...）
PANEL_ROOT = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsPairs/PairsImages"

# 可視化輸出的檔名（存在每個 scene 資料夾底下）
VIS_FILENAME = "vis_rgb_ir_pairs.jpg"

# 一些畫圖用設定
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS_BOX = 2
THICKNESS_TEXT = 1
THICKNESS_LINE = 2

COLOR_BOX_RGB = (0, 255, 0)   # RGB bbox 顏色 (B, G, R)
COLOR_BOX_IR  = (255, 0, 0)   # IR bbox 顏色
COLOR_LINE    = (0, 255, 255) # 連線顏色


def draw_panel_bbox_and_label(img, xyxy, label, color):
    """在影像上畫 panel 的 bbox 與文字標籤。"""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS_BOX)
    text_pos = (x1, max(0, y1 - 5))
    cv2.putText(img, label, text_pos, FONT, FONT_SCALE, color, THICKNESS_TEXT, cv2.LINE_AA)


def visualize_scene(scene_dir: str):
    """
    對單一場景做可視化：
      - scene_dir: e.g. ".../SolarPanelsPairs/PairsImages/1-1"
    """
    match_path = os.path.join(scene_dir, "match.json")
    if not os.path.isfile(match_path):
        print(f"[警告] 找不到 match.json：{match_path}，略過")
        return

    with open(match_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta       = data.get("meta", {})
    pairs      = data.get("pairs", [])
    id_map     = data.get("id_map", {})
    rgb_panels = data.get("rgb_panels", [])
    ir_panels  = data.get("ir_panels", [])

    if not rgb_panels or not ir_panels or not pairs:
        print(f"[警告] {scene_dir}：rgb_panels / ir_panels / pairs 資料不足，略過")
        return

    rgb_path = meta.get("rgb_path")
    ir_path  = meta.get("ir_path")
    if not rgb_path or not ir_path:
        print(f"[警告] {scene_dir} 的 meta 中沒有 rgb_path / ir_path，略過")
        return

    rgb_img = cv2.imread(rgb_path)
    ir_img  = cv2.imread(ir_path)

    if rgb_img is None or ir_img is None:
        print(f"[警告] 無法讀取 RGB/IR 圖片：{rgb_path}, {ir_path}，略過")
        return

    h_rgb, w_rgb, _ = rgb_img.shape
    h_ir,  w_ir,  _ = ir_img.shape

    # === 建立畫布：左邊 IR（縮小版，垂直置中），右邊 RGB（原尺寸） ===
    # 1. RGB 不縮放，直接用原尺寸
    rgb_resized = rgb_img.copy()
    h_rgb_r, w_rgb_r = h_rgb, w_rgb

    # 2. IR 縮放（高度 = RGB 高度 * shrink_ratio）
    shrink_ratio = 0.6
    target_h_ir = int(h_rgb_r * shrink_ratio)
    scale_ir = target_h_ir / h_ir
    w_ir_r = int(w_ir * scale_ir)
    ir_resized = cv2.resize(ir_img, (w_ir_r, target_h_ir))

    # 3. 畫布高度 = RGB 高度，寬度 = IR寬 + RGB寬
    canvas_h = h_rgb_r
    canvas_w = w_ir_r + w_rgb_r
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 4. IR 垂直置中
    ir_top = (canvas_h - target_h_ir) // 2
    canvas[ir_top:ir_top + target_h_ir, 0:w_ir_r, :] = ir_resized

    # 5. RGB 放在右側（貼上方，因為本來就滿高）
    canvas[0:h_rgb_r, w_ir_r:w_ir_r + w_rgb_r, :] = rgb_resized

    # === 對每個配對畫 bbox + 連線 (IR 左 → RGB 右) ===
    for p in pairs:
        rgb_idx = p.get("rgb_idx")
        ir_idx  = p.get("ir_idx")

        if rgb_idx is None or ir_idx is None:
            continue
        if rgb_idx >= len(rgb_panels) or ir_idx >= len(ir_panels):
            continue

        # id_map 在 json 裡 key 會變成字串，所以要做兩種查法
        panel_id = id_map.get(str(rgb_idx), id_map.get(rgb_idx))
        if panel_id is None:
            panel_id = f"idx{rgb_idx}"

        rgb_xyxy = rgb_panels[rgb_idx]["xyxy"]
        ir_xyxy  = ir_panels[ir_idx]["xyxy"]

        # RGB 不縮放，直接用原始座標
        x1r, y1r, x2r, y2r = rgb_xyxy
        # IR 縮放到 scale_ir
        x1i, y1i, x2i, y2i = ir_xyxy
        x1i *= scale_ir
        y1i *= scale_ir
        x2i *= scale_ir
        y2i *= scale_ir

        # 左半邊畫 IR panel（無水平 offset）
        draw_panel_bbox_and_label(
            canvas,
            (x1i, y1i + ir_top, x2i, y2i + ir_top),
            panel_id,
            COLOR_BOX_IR,
        )

        # 右半邊畫 RGB panel（x 要加上 w_ir_r 的 offset）
        draw_panel_bbox_and_label(
            canvas,
            (x1r + w_ir_r, y1r, x2r + w_ir_r, y2r),
            panel_id,
            COLOR_BOX_RGB,
        )

        # 計算中心點並畫連線：IR 中心 → RGB 中心
        cx_ir  = (x1i + x2i) / 2.0
        cy_ir  = (y1i + y2i) / 2.0 + ir_top
        cx_rgb = (x1r + x2r) / 2.0 + w_ir_r
        cy_rgb = (y1r + y2r) / 2.0

        cv2.line(
            canvas,
            (int(cx_ir),  int(cy_ir)),
            (int(cx_rgb), int(cy_rgb)),
            COLOR_LINE,
            THICKNESS_LINE,
            cv2.LINE_AA,
        )

    # === 輸出 ===
    out_path = os.path.join(scene_dir, VIS_FILENAME)
    cv2.imwrite(out_path, canvas)
    print(f"[INFO] 產生可視化圖：{out_path}")


def main():
    scenes = [
        d for d in os.listdir(PANEL_ROOT)
        if os.path.isdir(os.path.join(PANEL_ROOT, d))
    ]
    scenes.sort()

    print(f"[INFO] 在 {PANEL_ROOT} 發現 {len(scenes)} 個場景資料夾")

    for scene in scenes:
        scene_dir = os.path.join(PANEL_ROOT, scene)
        print(f"[INFO] 處理場景：{scene}")
        visualize_scene(scene_dir)

    print("[INFO] 全部場景可視化完成。")


if __name__ == "__main__":
    main()
