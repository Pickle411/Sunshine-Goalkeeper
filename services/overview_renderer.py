import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


@dataclass
class OverviewPaths:
    run_dir: str
    stage1_dir: str
    stage2_dir: str
    overview_dir: str
    overview_image_path: str
    overview_data_json_path: str


@dataclass
class OverviewResult:
    scene_name: str
    meta: Dict[str, Any]
    panels: List[Dict[str, Any]]
    paths: OverviewPaths

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OverviewRenderer:
    """
    第三階段：
    1. 讀取 stage1 的 match.json
    2. 讀取 stage2 的 panel_predictions.json
    3. 把 panel-level defect bbox 映射回整張 RGB 原圖
    4. 輸出 overview 圖與 UI 可用的 panel metadata
    """

    def run_from_run_dir(self, run_dir: str | Path) -> OverviewResult:
        run_dir = Path(run_dir)
        stage1_dir = run_dir / config.STAGE1_SUBDIR
        stage2_dir = run_dir / "stage2"

        match_json_path = run_dir / config.JSON_SUBDIR / config.MATCH_JSON_NAME
        panel_pred_json_path = stage2_dir / "json" / "panel_predictions.json"

        if not match_json_path.is_file():
            raise FileNotFoundError(f"找不到 stage1 match.json: {match_json_path}")
        if not panel_pred_json_path.is_file():
            raise FileNotFoundError(f"找不到 stage2 panel_predictions.json: {panel_pred_json_path}")

        with match_json_path.open("r", encoding="utf-8") as f:
            match_info = json.load(f)

        with panel_pred_json_path.open("r", encoding="utf-8") as f:
            panel_pred_info = json.load(f)

        scene_name = match_info.get("meta", {}).get("scene_name", run_dir.name)
        rgb_path = match_info.get("meta", {}).get("rgb_path")
        if not rgb_path:
            raise RuntimeError("match.json 內缺少 rgb_path")

        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise RuntimeError(f"無法讀取 RGB 原圖: {rgb_path}")

        overview_dir = stage2_dir / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)

        overview_image_path = overview_dir / "overview_rgb.jpg"
        overview_data_json_path = overview_dir / "overview_data.json"

        panel_predictions = panel_pred_info.get("panel_predictions", [])
        rendered_panels = self._build_panel_render_data(panel_predictions)

        overview_img = rgb_img.copy()
        for panel in rendered_panels:
            self._draw_panel_on_overview(overview_img, panel)

        cv2.imwrite(str(overview_image_path), overview_img)

        result = OverviewResult(
            scene_name=scene_name,
            meta={
                "scene_name": scene_name,
                "run_dir": str(run_dir),
                "stage1_dir": str(stage1_dir),
                "stage2_dir": str(stage2_dir),
                "rgb_path": rgb_path,
                "num_panels": len(rendered_panels),
                "num_defective_panels": sum(1 for p in rendered_panels if p["num_defects"] > 0),
                "num_normal_panels": sum(1 for p in rendered_panels if p["num_defects"] == 0),
            },
            panels=rendered_panels,
            paths=OverviewPaths(
                run_dir=str(run_dir),
                stage1_dir=str(stage1_dir),
                stage2_dir=str(stage2_dir),
                overview_dir=str(overview_dir),
                overview_image_path=str(overview_image_path),
                overview_data_json_path=str(overview_data_json_path),
            ),
        )

        self._write_json(overview_data_json_path, result.to_dict())
        return result

    def _build_panel_render_data(self, panel_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rendered_panels: List[Dict[str, Any]] = []

        for pred in panel_predictions:
            panel_id = pred.get("panel_id", "unknown")
            rgb_xyxy = pred.get("rgb_xyxy")
            if not rgb_xyxy:
                continue

            rgb_crop_path = pred.get("rgb_crop_path")
            ir_crop_path = pred.get("ir_crop_path")
            vis_path = pred.get("vis_path")
            ir_vis_path = pred.get("ir_vis_path")
            detections = pred.get("detections", [])
            num_defects = pred.get("num_defects", 0)
            status = pred.get("status", "unknown")

            panel_bbox = [float(v) for v in rgb_xyxy]
            mapped_detections = self._map_panel_detections_to_overview(
                panel_bbox=panel_bbox,
                rgb_crop_path=rgb_crop_path,
                detections=detections,
            )

            rendered_panels.append(
            {
                "panel_id": panel_id,
                "rgb_idx": pred.get("rgb_idx"),
                "ir_idx": pred.get("ir_idx"),
                "rgb_crop_path": rgb_crop_path,
                "ir_crop_path": ir_crop_path,
                "vis_path": vis_path,
                "ir_vis_path": ir_vis_path,
                "rgb_xyxy": panel_bbox,
                "ir_xyxy": pred.get("ir_xyxy"),
                "num_defects": num_defects,
                "status": status,
                "detections_panel": detections,
                "detections_overview": mapped_detections,
                "is_defective": num_defects > 0,
            }
        )

        rendered_panels.sort(key=lambda x: x["panel_id"])
        return rendered_panels

    def _map_panel_detections_to_overview(
        self,
        panel_bbox: List[float],
        rgb_crop_path: Optional[str],
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []

        if not rgb_crop_path or not detections:
            return mapped

        crop_img = cv2.imread(rgb_crop_path)
        if crop_img is None:
            return mapped

        crop_h, crop_w = crop_img.shape[:2]
        x1_panel, y1_panel, x2_panel, y2_panel = panel_bbox
        panel_w = max(x2_panel - x1_panel, 1.0)
        panel_h = max(y2_panel - y1_panel, 1.0)

        sx = panel_w / crop_w
        sy = panel_h / crop_h

        for det in detections:
            bbox = det.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue

            dx1, dy1, dx2, dy2 = bbox
            ox1 = x1_panel + dx1 * sx
            oy1 = y1_panel + dy1 * sy
            ox2 = x1_panel + dx2 * sx
            oy2 = y1_panel + dy2 * sy

            mapped.append(
                {
                    "bbox_xyxy": [float(ox1), float(oy1), float(ox2), float(oy2)],
                    "confidence": det.get("confidence"),
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                }
            )

        return mapped

    def _draw_panel_on_overview(self, img: np.ndarray, panel: Dict[str, Any]) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in panel["rgb_xyxy"]]
        panel_id = panel["panel_id"]
        num_defects = panel["num_defects"]

        # 有瑕疵 -> 紅色；正常 -> 綠色
        color = (0, 0, 255) if num_defects > 0 else (0, 255, 0)
        alpha = 0.28 if num_defects > 0 else 0.18

        # 半透明色塊覆蓋
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

        # 保留細框，讓模組邊界更清楚
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # 標籤
        label = f"{panel_id}"
        if num_defects > 0:
            label += f" | defect:{num_defects}"
        self._draw_label(img, label, (x1, max(y1 - 8, 0)), color)


    def _draw_label(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        y = max(y, th + 2)

        cv2.rectangle(
            img,
            (x, y - th - baseline - 4),
            (x + tw + 4, y + 2),
            color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x + 2, y - 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def run_overview_renderer(run_dir: str | Path) -> Dict[str, Any]:
    renderer = OverviewRenderer()
    result = renderer.run_from_run_dir(run_dir)
    return result.to_dict()


if __name__ == "__main__":
    sample_run_dir = "/mnt/shared/chuanyu_m11317028/SolarPanel/UI/data/runs/20260314_173820_sample_1-1"
    result = run_overview_renderer(sample_run_dir)
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))
    print(f"overview image -> {result['paths']['overview_image_path']}")
    print(f"overview data  -> {result['paths']['overview_data_json_path']}")