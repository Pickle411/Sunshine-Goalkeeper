import json
from pathlib import Path
from typing import Any, Dict, List

import config


class ResultFormatter:
    """
    把 stage1 / stage2 / overview 的輸出整理成 UI 容易直接使用的格式。
    """

    def build_ui_payload_from_run_dir(self, run_dir: str | Path) -> Dict[str, Any]:
        run_dir = Path(run_dir)

        match_json_path = run_dir / config.JSON_SUBDIR / config.MATCH_JSON_NAME
        panel_pred_json_path = run_dir / "stage2" / "json" / "panel_predictions.json"
        overview_json_path = run_dir / "stage2" / "overview" / "overview_data.json"

        if not match_json_path.is_file():
            raise FileNotFoundError(f"找不到 stage1 match.json: {match_json_path}")
        if not panel_pred_json_path.is_file():
            raise FileNotFoundError(f"找不到 stage2 panel_predictions.json: {panel_pred_json_path}")
        if not overview_json_path.is_file():
            raise FileNotFoundError(f"找不到 overview_data.json: {overview_json_path}")

        with match_json_path.open("r", encoding="utf-8") as f:
            match_info = json.load(f)

        with panel_pred_json_path.open("r", encoding="utf-8") as f:
            panel_pred_info = json.load(f)

        with overview_json_path.open("r", encoding="utf-8") as f:
            overview_info = json.load(f)

        panels = overview_info.get("panels", [])
        summary = self._build_summary(match_info, panel_pred_info, panels)
        panel_list = self._build_panel_list(panels)

        return {
            "scene_name": overview_info.get("scene_name"),
            "summary": summary,
            "overview": {
                "image_path": overview_info["paths"]["overview_image_path"],
                "data_path": overview_info["paths"]["overview_data_json_path"],
            },
            "panels": panel_list,
            "raw_paths": {
                "match_json_path": str(match_json_path),
                "panel_predictions_json_path": str(panel_pred_json_path),
                "overview_json_path": str(overview_json_path),
            },
        }

    def _build_summary(
        self,
        match_info: Dict[str, Any],
        panel_pred_info: Dict[str, Any],
        panels: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total_panels = len(panels)
        defective_panels = sum(1 for p in panels if p.get("num_defects", 0) > 0)
        normal_panels = total_panels - defective_panels

        total_detections = 0
        class_count: Dict[str, int] = {}

        for p in panels:
            for det in p.get("detections_panel", []):
                total_detections += 1
                cls_name = det.get("class_name")
                if cls_name is None:
                    cls_name = f"class_{det.get('class_id', -1)}"
                class_count[cls_name] = class_count.get(cls_name, 0) + 1

        return {
            "scene_name": match_info.get("meta", {}).get("scene_name"),
            "num_rgb_panels": match_info.get("meta", {}).get("num_rgb", 0),
            "num_ir_panels": match_info.get("meta", {}).get("num_ir", 0),
            "num_matched_panels": match_info.get("meta", {}).get("num_matched", 0),
            "num_panel_predictions": panel_pred_info.get("meta", {}).get("num_panel_predictions", 0),
            "total_panels": total_panels,
            "defective_panels": defective_panels,
            "normal_panels": normal_panels,
            "total_detections": total_detections,
            "defect_type_count": class_count,
        }

    def _build_panel_list(self, panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        panel_list: List[Dict[str, Any]] = []

        for p in panels:
            defect_types = []
            for det in p.get("detections_panel", []):
                cls_name = det.get("class_name")
                if cls_name is None:
                    cls_name = f"class_{det.get('class_id', -1)}"
                defect_types.append(cls_name)

            panel_list.append(
                {
                    "panel_id": p.get("panel_id"),
                    "is_defective": p.get("is_defective", False),
                    "num_defects": p.get("num_defects", 0),
                    "defect_types": defect_types,
                    "rgb_xyxy": p.get("rgb_xyxy"),
                    "rgb_crop_path": p.get("rgb_crop_path"),
                    "ir_crop_path": p.get("ir_crop_path"),
                    "vis_path": p.get("vis_path"),
                    "ir_vis_path": p.get("ir_vis_path"),
                    "detections_panel": p.get("detections_panel", []),
                    "detections_overview": p.get("detections_overview", []),
                    "status": p.get("status"),
                }
            )

        panel_list.sort(key=lambda x: x["panel_id"])
        return panel_list


def build_ui_payload(run_dir: str | Path) -> Dict[str, Any]:
    formatter = ResultFormatter()
    return formatter.build_ui_payload_from_run_dir(run_dir)


if __name__ == "__main__":
    sample_run_dir = "/mnt/shared/chuanyu_m11317028/SolarPanel/UI/data/runs/20260314_173820_sample_1-1"
    payload = build_ui_payload(sample_run_dir)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"overview image -> {payload['overview']['image_path']}")
    print(f"num panels     -> {len(payload['panels'])}")