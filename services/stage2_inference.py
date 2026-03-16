import json
import shutil
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config

warnings.filterwarnings("ignore")

# 優先使用你專案裡的 ultralytics
if str(config.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PROJECT_ROOT))

from ultralytics import YOLO


# ===== 類別分組：沿用你提供的 predict_visualize_conf =====
IR_CLASSES = {0, 1}
VIS_CLASSES = {2, 3}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Stage2Paths:
    run_dir: str
    stage1_dir: str
    stage2_dir: str
    stage2_dataset_dir: str
    visible_dir: str
    infrared_dir: str
    pred_visible_dir: str
    pred_infrared_dir: str
    json_dir: str
    panel_result_json_path: str


@dataclass
class PanelPrediction:
    panel_id: str
    rgb_idx: Optional[int]
    ir_idx: Optional[int]
    rgb_crop_path: Optional[str]
    ir_crop_path: Optional[str]
    rgb_xyxy: Optional[List[float]]
    ir_xyxy: Optional[List[float]]
    num_defects: int
    detections: List[Dict[str, Any]]
    vis_path: Optional[str]
    ir_vis_path: Optional[str]
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Stage2Result:
    scene_name: str
    meta: Dict[str, Any]
    panel_predictions: List[PanelPrediction]
    paths: Stage2Paths

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Stage2Inference:
    """
    第二階段：完全對齊你提供的 predict_visualize_conf 邏輯
    1. 建立臨時資料集：
       test/visible/images
       test/infrared/images
    2. 以 visible/images 作為 source
    3. 用 use_simotm="RGBRGB6C" + channels=6 自動配對 infrared/images
    4. 從 results 直接解析 boxes
    5. 產生 pred_visible / pred_infrared 與 panel_predictions.json
    """

    def __init__(
        self,
        model_path: Path = config.ANOMALY_MODEL_PATH,
        device: str = config.DEVICE,
        imgsz: int = 512,
        batch: int = 4,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = imgsz
        self.batch = batch
        self.conf = conf
        self.iou = iou
        self._model: Optional[YOLO] = None

    # =========================================================
    # Public API
    # =========================================================
    def run_from_stage1_result(self, stage1_result: Dict[str, Any]) -> Stage2Result:
        scene_name = stage1_result.get("scene_name", "unknown_scene")
        crop_pairs = stage1_result.get("crop_pairs", [])
        paths_info = stage1_result.get("paths", {})

        run_dir = Path(paths_info["run_dir"])
        stage1_dir = Path(paths_info["stage1_dir"])

        stage2_paths = self._prepare_stage2_dirs(run_dir=run_dir, stage1_dir=stage1_dir)
        pair_map = self._build_temp_dataset(
            stage2_dataset_dir=Path(stage2_paths.stage2_dataset_dir),
            crop_pairs=crop_pairs,
        )

        results = self._run_predict(stage2_paths)
        panel_predictions = self._build_panel_predictions(
            results=results,
            pair_map=pair_map,
            stage2_paths=stage2_paths,
        )

        result = Stage2Result(
            scene_name=scene_name,
            meta={
                "scene_name": scene_name,
                "run_dir": str(run_dir),
                "stage1_dir": str(stage1_dir),
                "stage2_dir": stage2_paths.stage2_dir,
                "model_path": str(self.model_path),
                "imgsz": self.imgsz,
                "batch": self.batch,
                "conf": self.conf,
                "iou": self.iou,
                "num_pairs": len(crop_pairs),
                "num_panel_predictions": len(panel_predictions),
            },
            panel_predictions=panel_predictions,
            paths=stage2_paths,
        )

        self._write_json(Path(stage2_paths.panel_result_json_path), result.to_dict())
        return result

    def run_from_run_dir(self, run_dir: str | Path) -> Stage2Result:
        run_dir = Path(run_dir)
        match_json_path = run_dir / config.JSON_SUBDIR / config.MATCH_JSON_NAME
        stage1_dir = run_dir / config.STAGE1_SUBDIR

        if not match_json_path.is_file():
            raise FileNotFoundError(f"找不到 stage1 的 match.json：{match_json_path}")

        with match_json_path.open("r", encoding="utf-8") as f:
            match_info = json.load(f)

        crop_pairs = self._collect_crop_pairs_from_match_info(stage1_dir, match_info)

        stage2_paths = self._prepare_stage2_dirs(run_dir=run_dir, stage1_dir=stage1_dir)
        pair_map = self._build_temp_dataset(
            stage2_dataset_dir=Path(stage2_paths.stage2_dataset_dir),
            crop_pairs=crop_pairs,
        )

        results = self._run_predict(stage2_paths)
        panel_predictions = self._build_panel_predictions(
            results=results,
            pair_map=pair_map,
            stage2_paths=stage2_paths,
        )

        scene_name = match_info.get("meta", {}).get("scene_name", run_dir.name)

        result = Stage2Result(
            scene_name=scene_name,
            meta={
                "scene_name": scene_name,
                "run_dir": str(run_dir),
                "stage1_dir": str(stage1_dir),
                "stage2_dir": stage2_paths.stage2_dir,
                "model_path": str(self.model_path),
                "imgsz": self.imgsz,
                "batch": self.batch,
                "conf": self.conf,
                "iou": self.iou,
                "num_pairs": len(crop_pairs),
                "num_panel_predictions": len(panel_predictions),
            },
            panel_predictions=panel_predictions,
            paths=stage2_paths,
        )

        self._write_json(Path(stage2_paths.panel_result_json_path), result.to_dict())
        return result

    # =========================================================
    # Build temporary paired dataset
    # =========================================================
    def _build_temp_dataset(
        self,
        stage2_dataset_dir: Path,
        crop_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        建立符合你原本 predict 流程的資料結構：
          test/visible/images
          test/infrared/images

        並統一 resize 成 512x512，避免 visible/infrared 尺寸不一致。
        """
        if stage2_dataset_dir.exists():
            shutil.rmtree(stage2_dataset_dir)

        visible_dir = stage2_dataset_dir / "test" / "visible" / "images"
        infrared_dir = stage2_dataset_dir / "test" / "infrared" / "images"
        visible_dir.mkdir(parents=True, exist_ok=True)
        infrared_dir.mkdir(parents=True, exist_ok=True)

        pair_map: Dict[str, Dict[str, Any]] = {}
        target_size = (512, 512)  # width, height

        for pair in crop_pairs:
            panel_id = pair.get("panel_id")
            rgb_crop_path = pair.get("rgb_crop_path")
            ir_crop_path = pair.get("ir_crop_path")

            if not panel_id or not rgb_crop_path or not ir_crop_path:
                continue

            rgb_img = cv2.imread(str(rgb_crop_path), cv2.IMREAD_COLOR)
            ir_img = cv2.imread(str(ir_crop_path), cv2.IMREAD_COLOR)
            if rgb_img is None or ir_img is None:
                continue

            rgb_img = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_LINEAR)
            ir_img = cv2.resize(ir_img, target_size, interpolation=cv2.INTER_LINEAR)

            vis_dst = visible_dir / f"{panel_id}.jpg"
            ir_dst = infrared_dir / f"{panel_id}.jpg"

            cv2.imwrite(str(vis_dst), rgb_img)
            cv2.imwrite(str(ir_dst), ir_img)

            pair_map[f"{panel_id}.jpg"] = {
                "panel_id": panel_id,
                "rgb_idx": pair.get("rgb_idx"),
                "ir_idx": pair.get("ir_idx"),
                "rgb_crop_path": pair.get("rgb_crop_path"),
                "ir_crop_path": pair.get("ir_crop_path"),
                "rgb_xyxy": pair.get("rgb_xyxy"),
                "ir_xyxy": pair.get("ir_xyxy"),
                "temp_visible_path": str(vis_dst),
                "temp_infrared_path": str(ir_dst),
            }

        return pair_map

    # =========================================================
    # Predict: exactly follow your provided script
    # =========================================================
    def _run_predict(self, stage2_paths: Stage2Paths):
        model = self._load_model()
        vis_dir = Path(stage2_paths.visible_dir)

        results = model.predict(
            source=str(vis_dir),
            imgsz=self.imgsz,
            batch=self.batch,
            conf=self.conf,
            iou=self.iou,
            save=False,
            use_simotm="RGBRGB6C",
            channels=6,
            project=str(Path(stage2_paths.stage2_dir) / "predict_runs"),
            name="inference",
            show=False,
            verbose=True,
            device=self.device,
        )
        return results

    # =========================================================
    # Parse predictions from results directly
    # =========================================================
    def _build_panel_predictions(
        self,
        results,
        pair_map: Dict[str, Dict[str, Any]],
        stage2_paths: Stage2Paths,
    ) -> List[PanelPrediction]:
        pred_visible_dir = Path(stage2_paths.pred_visible_dir)
        pred_infrared_dir = Path(stage2_paths.pred_infrared_dir)
        pred_visible_dir.mkdir(parents=True, exist_ok=True)
        pred_infrared_dir.mkdir(parents=True, exist_ok=True)

        class_names = self._get_class_names()
        panel_predictions: List[PanelPrediction] = []

        for r in results:
            fname = Path(r.path).name
            pair_info = pair_map.get(fname)
            if pair_info is None:
                continue

            panel_id = pair_info["panel_id"]

            # 讀原始 crop，不是讀 512x512 臨時圖
            orig_rgb_path = pair_info.get("rgb_crop_path")
            orig_ir_path = pair_info.get("ir_crop_path")

            vis_img = cv2.imread(orig_rgb_path, cv2.IMREAD_COLOR) if orig_rgb_path else None
            ir_img = cv2.imread(orig_ir_path, cv2.IMREAD_COLOR) if orig_ir_path else None

            if vis_img is None or ir_img is None:
                panel_predictions.append(
                    PanelPrediction(
                        panel_id=panel_id,
                        rgb_idx=pair_info.get("rgb_idx"),
                        ir_idx=pair_info.get("ir_idx"),
                        rgb_crop_path=pair_info.get("rgb_crop_path"),
                        ir_crop_path=pair_info.get("ir_crop_path"),
                        rgb_xyxy=pair_info.get("rgb_xyxy"),
                        ir_xyxy=pair_info.get("ir_xyxy"),
                        num_defects=0,
                        detections=[],
                        vis_path=None,
                        ir_vis_path=None,
                        status="failed",
                        error="無法讀取原始 RGB/IR crop",
                    )
                )
                continue

            # 模型輸出是針對 512x512 臨時圖，所以要映回原始 crop 尺寸
            temp_vis_path = pair_info.get("temp_visible_path")
            temp_ir_path = pair_info.get("temp_infrared_path")

            temp_vis_img = cv2.imread(temp_vis_path, cv2.IMREAD_COLOR) if temp_vis_path else None
            temp_ir_img = cv2.imread(temp_ir_path, cv2.IMREAD_COLOR) if temp_ir_path else None

            if temp_vis_img is None or temp_ir_img is None:
                panel_predictions.append(
                    PanelPrediction(
                        panel_id=panel_id,
                        rgb_idx=pair_info.get("rgb_idx"),
                        ir_idx=pair_info.get("ir_idx"),
                        rgb_crop_path=pair_info.get("rgb_crop_path"),
                        ir_crop_path=pair_info.get("ir_crop_path"),
                        rgb_xyxy=pair_info.get("rgb_xyxy"),
                        ir_xyxy=pair_info.get("ir_xyxy"),
                        num_defects=0,
                        detections=[],
                        vis_path=None,
                        ir_vis_path=None,
                        status="failed",
                        error="無法讀取臨時 visible/infrared 圖",
                    )
                )
                continue

            orig_h_rgb, orig_w_rgb = vis_img.shape[:2]
            orig_h_ir, orig_w_ir = ir_img.shape[:2]
            temp_h_rgb, temp_w_rgb = temp_vis_img.shape[:2]
            temp_h_ir, temp_w_ir = temp_ir_img.shape[:2]

            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
            cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else np.empty((0,))
            conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.zeros((len(cls),))

            detections = self._build_detection_list(xyxy, cls, conf, class_names)

            # 先複製原圖，確保就算沒有瑕疵也會保留完整模組圖
            vis_vis = vis_img.copy()
            ir_vis = ir_img.copy()

            # 分模態畫框：0,1 -> IR；2,3 -> RGB
            for i, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, cls)):
                cid = int(c)
                conf_val = float(conf[i]) if len(conf) > i else 0.0

                if cid in VIS_CLASSES:
                    sx = orig_w_rgb / temp_w_rgb
                    sy = orig_h_rgb / temp_h_rgb
                    rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                    self._draw_single_box(vis_vis, rx1, ry1, rx2, ry2, cid, conf_val)

                elif cid in IR_CLASSES:
                    sx = orig_w_ir / temp_w_ir
                    sy = orig_h_ir / temp_h_ir
                    rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                    self._draw_single_box(ir_vis, rx1, ry1, rx2, ry2, cid, conf_val)

            vis_out = pred_visible_dir / fname
            ir_out = pred_infrared_dir / fname
            cv2.imwrite(str(vis_out), vis_vis)
            cv2.imwrite(str(ir_out), ir_vis)

            panel_predictions.append(
                PanelPrediction(
                    panel_id=panel_id,
                    rgb_idx=pair_info.get("rgb_idx"),
                    ir_idx=pair_info.get("ir_idx"),
                    rgb_crop_path=pair_info.get("rgb_crop_path"),
                    ir_crop_path=pair_info.get("ir_crop_path"),
                    rgb_xyxy=pair_info.get("rgb_xyxy"),
                    ir_xyxy=pair_info.get("ir_xyxy"),
                    num_defects=len(detections),
                    detections=detections,
                    vis_path=str(vis_out),
                    ir_vis_path=str(ir_out),
                    status="success",
                    error=None,
                )
            )

        panel_predictions.sort(key=lambda x: x.panel_id)
        return panel_predictions

    def _build_detection_list(
        self,
        xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        class_names: List[str],
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        for i, box in enumerate(xyxy):
            cls_id = int(cls[i])
            conf_val = float(conf[i]) if len(conf) > i else None
            x1, y1, x2, y2 = box.tolist()

            class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"

            detections.append(
                {
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf_val,
                    "class_id": cls_id,
                    "class_name": class_name,
                }
            )

        return detections

    def _draw_boxes(
        self,
        img: np.ndarray,
        boxes_xyxy: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        cls_names: List[str],
        keep_cls: set,
    ) -> np.ndarray:
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes_xyxy, classes)):
            cid = int(c)
            if cid not in keep_cls:
                continue

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{cid}:{confs[i]:.2f}"

            box_color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            bg_x1 = x1
            bg_y1 = max(0, y1 - th - baseline - 6)
            bg_x2 = x1 + tw + 6
            bg_y2 = y1

            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)

            text_y = max(th + 2, y1 - 4)
            cv2.putText(
                img,
                label,
                (x1 + 3, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return img

    def _draw_single_box(
        self,
        img: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cid: int,
        conf_val: float,
    ) -> None:
        h, w = img.shape[:2]
        base = min(h, w)

        # 字更小、線更細
        font_scale = max(0.22, min(0.45, base / 420.0))
        thickness = 1
        box_thickness = 1
        pad = 1

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{cid}:{conf_val:.2f}" 
        # label = f"{cid}" # 若不要信心值

        box_color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        bg_x1 = x1
        bg_y1 = max(0, y1 - th - baseline - 2)
        bg_x2 = x1 + tw + 2
        bg_y2 = y1

        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)

        text_y = max(th, y1 - 1)
        cv2.putText(
            img,
            label,
            (x1 + 1, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # =========================================================
    # Helpers
    # =========================================================
    def _load_model(self) -> YOLO:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"找不到 anomaly 模型權重：{self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def _get_class_names(self) -> List[str]:
        model = self._load_model()
        names = getattr(model.model, "names", None)

        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys())]
        if isinstance(names, list):
            return [str(x) for x in names]

        return [
            "ir_abnormal",
            "ir_abnormal_piece",
            "rgb_module_abnormal_wp",
            "rgb_module_abnormal_sun",
        ]

    def _prepare_stage2_dirs(self, run_dir: Path, stage1_dir: Path) -> Stage2Paths:
        stage2_dir = run_dir / "stage2"
        stage2_dataset_dir = run_dir / "stage2_dataset"
        visible_dir = stage2_dataset_dir / "test" / "visible" / "images"
        infrared_dir = stage2_dataset_dir / "test" / "infrared" / "images"
        pred_visible_dir = stage2_dir / "pred_visible"
        pred_infrared_dir = stage2_dir / "pred_infrared"
        json_dir = stage2_dir / "json"
        panel_result_json_path = json_dir / "panel_predictions.json"

        for d in [stage2_dir, stage2_dataset_dir, pred_visible_dir, pred_infrared_dir, json_dir]:
            d.mkdir(parents=True, exist_ok=True)

        return Stage2Paths(
            run_dir=str(run_dir),
            stage1_dir=str(stage1_dir),
            stage2_dir=str(stage2_dir),
            stage2_dataset_dir=str(stage2_dataset_dir),
            visible_dir=str(visible_dir),
            infrared_dir=str(infrared_dir),
            pred_visible_dir=str(pred_visible_dir),
            pred_infrared_dir=str(pred_infrared_dir),
            json_dir=str(json_dir),
            panel_result_json_path=str(panel_result_json_path),
        )

    def _collect_crop_pairs_from_match_info(
        self,
        stage1_dir: Path,
        match_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        pairs = match_info.get("pairs", [])
        id_map = match_info.get("id_map", {})
        rgb_panels = match_info.get("rgb_panels", [])
        ir_panels = match_info.get("ir_panels", [])

        rgb_dir = stage1_dir / "RGB"
        ir_dir = stage1_dir / "IR"

        crop_pairs = []
        for p in pairs:
            rgb_idx = p.get("rgb_idx")
            ir_idx = p.get("ir_idx")

            panel_id = id_map.get(str(rgb_idx), id_map.get(rgb_idx))
            if panel_id is None:
                panel_id = f"idx{rgb_idx}"

            rgb_path = self._find_image_by_stem(rgb_dir, panel_id)
            ir_path = self._find_image_by_stem(ir_dir, panel_id)

            rgb_xyxy = None
            ir_xyxy = None
            if rgb_idx is not None and isinstance(rgb_idx, int) and 0 <= rgb_idx < len(rgb_panels):
                rgb_xyxy = rgb_panels[rgb_idx].get("xyxy")
            if ir_idx is not None and isinstance(ir_idx, int) and 0 <= ir_idx < len(ir_panels):
                ir_xyxy = ir_panels[ir_idx].get("xyxy")

            crop_pairs.append(
                {
                    "panel_id": panel_id,
                    "rgb_idx": rgb_idx,
                    "ir_idx": ir_idx,
                    "rgb_crop_path": str(rgb_path) if rgb_path else None,
                    "ir_crop_path": str(ir_path) if ir_path else None,
                    "rgb_xyxy": rgb_xyxy,
                    "ir_xyxy": ir_xyxy,
                }
            )

        crop_pairs.sort(key=lambda x: x["panel_id"])
        return crop_pairs

    def _find_image_by_stem(self, folder: Path, stem: str) -> Optional[Path]:
        if not folder.exists():
            return None
        for suffix in IMG_EXTS:
            p = folder / f"{stem}{suffix}"
            if p.is_file():
                return p
        return None

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def run_stage2_from_stage1_result(stage1_result: Dict[str, Any]) -> Dict[str, Any]:
    runner = Stage2Inference()
    result = runner.run_from_stage1_result(stage1_result)
    return result.to_dict()


def run_stage2_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    runner = Stage2Inference()
    result = runner.run_from_run_dir(run_dir)
    return result.to_dict()


if __name__ == "__main__":
    sample_run_dir = "/mnt/shared/chuanyu_m11317028/SolarPanel/UI/data/runs/20260314_173820_sample_1-1"
    result = run_stage2_from_run_dir(sample_run_dir)
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))
    print(f"panel_predictions.json -> {result['paths']['panel_result_json_path']}")
