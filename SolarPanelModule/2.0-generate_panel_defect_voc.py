# 功能：
# 1. 讀取每個場景資料夾下的 match.json（由 1.0-run_pipeline.py 產生）
#    - 其中已包含 rgb_panels / ir_panels / pairs / id_map
# 2. 讀取場景 RGB / IR 整張圖的 VOC 瑕疵標註 (AllLabel/Voc)
# 3. 將落在某一片 panel 內的瑕疵 bbox 轉換到 panel 座標系
# 4. 對每一片 panel，輸出一個 VOC 檔：
#       <scene>/RGB/r1_c1.jpg 對應 <scene>/RGB/r1_c1.xml
#       <scene>/IR/r1_c1.jpg  對應 <scene>/IR/r1_c1.xml

import os
import sys
import cv2
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

PROJECT_ROOT = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT"
sys.path.append(PROJECT_ROOT)

# ================= 路徑設定 =================

# 場景原圖 (整張場域圖)
RGB_FULL_IMG_DIR = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsModule/RGB/AllImage"
IR_FULL_IMG_DIR  = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsModule/IR/AllImage"

# 整張圖的 VOC 瑕疵標註
RGB_FULL_VOC_DIR = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsPairs/RGB/AllLabel/Voc"
IR_FULL_VOC_DIR  = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsPairs/IR/AllLabel/Voc"

# panel 小圖所在位置 (1.0-run_pipeline 切出來的)
PANEL_ROOT = "/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT/dataset/SolarPanelsPairs/PairsImages"

# ====================================================


# ========== 工具：讀 / 寫 VOC ==========

def parse_voc(xml_path: str) -> List[Dict[str, Any]]:
    """
    讀取 VOC，回傳物件列表：
    [{"name": str, "xmin": int, "ymin": int, "xmax": int, "ymax": int}, ...]
    """
    if not os.path.isfile(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        objs.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
    return objs


def write_voc(
    xml_path: str,
    img_path: str,
    objects: List[Dict[str, Any]],
):
    """
    寫出 VOC xml（即使 objects 為空也會產生 xml）。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"寫 VOC 時讀不到圖片: {img_path}")
    h, w, c = img.shape

    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(img_path))

    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(img_path)

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text  = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text  = str(c)

    for obj in objects:
        obj_el = ET.SubElement(annotation, "object")
        name_el = ET.SubElement(obj_el, "name")
        name_el.text = obj["name"]

        bnd = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(obj["xmin"]))
        ET.SubElement(bnd, "ymin").text = str(int(obj["ymin"]))
        ET.SubElement(bnd, "xmax").text = str(int(obj["xmax"]))
        ET.SubElement(bnd, "ymax").text = str(int(obj["ymax"]))

    tree = ET.ElementTree(annotation)
    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    tree.write(xml_path, encoding="utf-8")


# ========== 工具：將整張圖的瑕疵映射到 panel ==========

def map_defects_to_panels(
    full_objects: List[Dict[str, Any]],
    panel_xyxy: tuple,
    panel_img_path: str,
) -> List[Dict[str, Any]]:
    """
    將整張圖的瑕疵物件映射到某片 panel：
    - 用 defect bbox 中心點判斷是否屬於該 panel
    - 座標轉成 panel 內相對位置
    """
    x1_p, y1_p, x2_p, y2_p = panel_xyxy
    x1_p, y1_p, x2_p, y2_p = map(float, (x1_p, y1_p, x2_p, y2_p))

    img = cv2.imread(panel_img_path)
    if img is None:
        raise RuntimeError(f"讀不到 panel 圖片: {panel_img_path}")
    ph, pw, _ = img.shape

    mapped = []
    for obj in full_objects:
        xmin = obj["xmin"]
        ymin = obj["ymin"]
        xmax = obj["xmax"]
        ymax = obj["ymax"]

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # 判斷中心點是否在 panel 裡
        if not (x1_p <= cx <= x2_p and y1_p <= cy <= y2_p):
            continue

        # 映射到 panel 座標系
        nxmin = max(0, xmin - x1_p)
        nymin = max(0, ymin - y1_p)
        nxmax = min(pw, xmax - x1_p)
        nymax = min(ph, ymax - y1_p)

        if nxmax <= nxmin or nymax <= nymin:
            continue

        mapped.append({
            "name": obj["name"],
            "xmin": int(nxmin),
            "ymin": int(nymin),
            "xmax": int(nxmax),
            "ymax": int(nymax),
        })

    return mapped


# ========== 主流程：逐場景處理 ==========

def main():
    # 場景名稱來自 panel_root 底下的資料夾，例如 1-1, 1-2...
    scenes = sorted(
        d for d in os.listdir(PANEL_ROOT)
        if os.path.isdir(os.path.join(PANEL_ROOT, d))
    )

    print(f"[INFO] 發現 {len(scenes)} 個場景資料夾")

    for scene in scenes:
        scene_dir = os.path.join(PANEL_ROOT, scene)
        print(f"[INFO] 處理場景：{scene}")

        # 讀 1.0-run_pipeline 產生的 match.json
        match_path = os.path.join(scene_dir, "match.json")
        if not os.path.isfile(match_path):
            print(f"[警告] 找不到 match.json：{match_path}，略過")
            continue

        with open(match_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta       = data.get("meta", {})
        pairs      = data.get("pairs", [])
        id_map     = data.get("id_map", {})
        rgb_panels = data.get("rgb_panels", [])
        ir_panels  = data.get("ir_panels", [])

        if not rgb_panels or not ir_panels or not pairs:
            print(f"[警告] {scene}：rgb_panels / ir_panels / pairs 資料不足，略過")
            continue

        # 原始整張圖 (路徑直接從 meta 拿，比較穩)
        rgb_full_img = meta.get("rgb_path")
        ir_full_img  = meta.get("ir_path")

        if not rgb_full_img or not os.path.isfile(rgb_full_img):
            # fallback: 用檔名推
            rgb_full_img = os.path.join(RGB_FULL_IMG_DIR, f"{scene}.JPG")
            if not os.path.isfile(rgb_full_img):
                rgb_full_img = os.path.join(RGB_FULL_IMG_DIR, f"{scene}.jpg")

        if not ir_full_img or not os.path.isfile(ir_full_img):
            ir_full_img = os.path.join(IR_FULL_IMG_DIR, f"{scene}.JPG")
            if not os.path.isfile(ir_full_img):
                ir_full_img = os.path.join(IR_FULL_IMG_DIR, f"{scene}.jpg")

        if not os.path.isfile(rgb_full_img) or not os.path.isfile(ir_full_img):
            print(f"[警告] 找不到原始 RGB/IR 圖片：{scene}")
            continue

        # 整張圖 VOC 瑕疵標註
        rgb_full_voc = os.path.join(RGB_FULL_VOC_DIR, f"{scene}.xml")
        ir_full_voc  = os.path.join(IR_FULL_VOC_DIR,  f"{scene}.xml")

        rgb_full_objs = parse_voc(rgb_full_voc)
        ir_full_objs  = parse_voc(ir_full_voc)

        if not rgb_full_objs:
            print(f"[警告] RGB 整張圖沒有 VOC 標註或檔案缺失：{rgb_full_voc}")
        if not ir_full_objs:
            print(f"[警告] IR 整張圖沒有 VOC 標註或檔案缺失：{ir_full_voc}")

        rgb_panel_dir = os.path.join(scene_dir, "RGB")
        ir_panel_dir  = os.path.join(scene_dir, "IR")

        # 逐 pair 把瑕疵映射到 panel
        for p in pairs:
            rgb_idx = p.get("rgb_idx")
            ir_idx  = p.get("ir_idx")

            # id_map 可能是字串 key，所以兩種都試
            panel_id = id_map.get(str(rgb_idx), id_map.get(rgb_idx))
            if panel_id is None:
                continue

            # 對應的小圖路徑
            rgb_panel_img = os.path.join(rgb_panel_dir, f"{panel_id}.jpg")
            ir_panel_img  = os.path.join(ir_panel_dir,  f"{panel_id}.jpg")

            if not os.path.isfile(rgb_panel_img) or not os.path.isfile(ir_panel_img):
                print(f"[警告] 找不到 panel 圖片：{scene} {panel_id}")
                continue

            # 取當初偵測時使用的 bbox（直接用 match.json 裡的）
            try:
                rgb_xyxy = rgb_panels[rgb_idx]["xyxy"]
                ir_xyxy  = ir_panels[ir_idx]["xyxy"]
            except IndexError:
                print(f"[警告] {scene}：rgb_idx/ir_idx 超出範圍，略過 {panel_id}")
                continue

            # RGB：整張圖瑕疵 → panel
            rgb_objs_panel = map_defects_to_panels(
                rgb_full_objs,
                rgb_xyxy,
                rgb_panel_img,
            )
            rgb_panel_xml = os.path.join(rgb_panel_dir, f"{panel_id}.xml")
            write_voc(rgb_panel_xml, rgb_panel_img, rgb_objs_panel)

            # IR：整張圖瑕疵 → panel
            ir_objs_panel = map_defects_to_panels(
                ir_full_objs,
                ir_xyxy,
                ir_panel_img,
            )
            ir_panel_xml = os.path.join(ir_panel_dir, f"{panel_id}.xml")
            write_voc(ir_panel_xml, ir_panel_img, ir_objs_panel)

        print(f"[INFO] 場景 {scene} 處理完成。")

    print("[INFO] 所有場景 VOC panel 標註完成。")


if __name__ == "__main__":
    main()
