import json
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

# 先讓 Python 優先吃到你專案裡的 ultralytics
if str(config.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PROJECT_ROOT))

if str(config.PANEL_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(config.PANEL_TOOL_ROOT))

import cv2
from ultralytics import YOLO
from match_panels import greedy_match, assign_row_col
from crop_panels import crop_matched_panels


@dataclass
class Stage1Paths:
    run_dir: str
    original_dir: str
    stage1_dir: str
    rgb_crop_dir: str
    ir_crop_dir: str
    json_dir: str
    match_json_path: str
    rgb_input_path: str
    ir_input_path: str


@dataclass
class Stage1Result:
    scene_name: str
    meta: Dict[str, Any]
    rgb_panels: List[Dict[str, Any]]
    ir_panels: List[Dict[str, Any]]
    pairs: List[Dict[str, Any]]
    id_map: Dict[str, str]
    crop_pairs: List[Dict[str, Any]]
    paths: Stage1Paths

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_panels_xyxy_ui(model: YOLO, img_path: str, conf_thres=0.5):
    """
    UI 專用 wrapper:
    邏輯沿用你原本 detect_panels.py，
    但不改原檔，也不刪 channels/use_simotm。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"無法讀取影像: {img_path}")
    H, W = img.shape[:2]

    # 不刪原本模型需要的設定，明確指定成模型1可接受的單模態 3-channel 推論
    if hasattr(model, "overrides"):
        model.overrides["channels"] = 3
        model.overrides["use_simotm"] = "RGB"

    res = model(
        img_path,
        conf=conf_thres,
        verbose=False,
        channels=3,
        use_simotm="RGB",
    )[0]

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return {"size": (H, W), "panels": []}

    xyxy_list = boxes.xyxy.cpu().numpy()

    panels = []
    for (x1, y1, x2, y2) in xyxy_list:
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        panels.append({
            "cx": float(cx),
            "cy": float(cy),
            "xyxy": (float(x1), float(y1), float(x2), float(y2))
        })

    return {"size": (H, W), "panels": panels}


class Stage1Pipeline:
    """
    Stage 1:
    1. 載入 IR / RGB panel detection model
    2. 偵測 panels
    3. greedy 配對
    4. assign row/col ID
    5. 裁切 matched panels
    6. 輸出 match.json 與結構化結果
    """

    def __init__(
        self,
        rgb_model_path: Path = config.MODULE_RGB_MODEL_PATH,
        ir_model_path: Path = config.MODULE_IR_MODEL_PATH,
        device: str = config.DEVICE,
        conf_thres: float = config.MODULE_CONF_THRES,
        row_eps: float = config.ROW_EPS,
        max_dist: float = config.MAX_DIST,
    ) -> None:
        self.rgb_model_path = Path(rgb_model_path)
        self.ir_model_path = Path(ir_model_path)
        self.device = device
        self.conf_thres = conf_thres
        self.row_eps = row_eps
        self.max_dist = max_dist

        self._rgb_model: Optional[YOLO] = None
        self._ir_model: Optional[YOLO] = None

    def run(
        self,
        rgb_input_path: str | Path,
        ir_input_path: str | Path,
        scene_name: Optional[str] = None,
        run_name: Optional[str] = None,
        copy_inputs: bool = True,
    ) -> Stage1Result:
        rgb_input_path = Path(rgb_input_path)
        ir_input_path = Path(ir_input_path)

        self._validate_input_image(rgb_input_path, "RGB")
        self._validate_input_image(ir_input_path, "IR")

        if scene_name is None:
            scene_name = rgb_input_path.stem

        paths = self._prepare_run_dirs(
            rgb_input_path=rgb_input_path,
            ir_input_path=ir_input_path,
            scene_name=scene_name,
            run_name=run_name,
            copy_inputs=copy_inputs,
        )

        rgb_model = self._load_rgb_model()
        ir_model = self._load_ir_model()

        # 這裡只把 detect 換成 UI wrapper
        # 配對 / 命名 / 裁切仍沿用你原本方法
        rgb_det = _detect_panels_xyxy_ui(
            rgb_model,
            str(paths.rgb_input_path),
            conf_thres=self.conf_thres,
        )
        ir_det = _detect_panels_xyxy_ui(
            ir_model,
            str(paths.ir_input_path),
            conf_thres=self.conf_thres,
        )

        rgb_panels = rgb_det.get("panels", [])
        ir_panels = ir_det.get("panels", [])

        if not rgb_panels or not ir_panels:
            raise RuntimeError(
                f"RGB 或 IR 沒有偵測到任何 panel。RGB={len(rgb_panels)}, IR={len(ir_panels)}"
            )

        matched_pairs = greedy_match(rgb_panels, ir_panels, max_dist=self.max_dist)
        if not matched_pairs:
            raise RuntimeError("panel 偵測成功，但 greedy_match 沒有配對到任何 panel。")

        id_map = assign_row_col(rgb_panels, row_eps=self.row_eps)

        crop_matched_panels(
            str(paths.rgb_input_path),
            str(paths.ir_input_path),
            rgb_panels,
            ir_panels,
            matched_pairs,
            id_map,
            str(paths.stage1_dir),
        )

        match_info = {
            "meta": {
                "scene_name": scene_name,
                "rgb_path": str(paths.rgb_input_path),
                "ir_path": str(paths.ir_input_path),
                "num_rgb": len(rgb_panels),
                "num_ir": len(ir_panels),
                "num_matched": len(matched_pairs),
                "conf_thres": self.conf_thres,
                "row_eps": self.row_eps,
                "max_dist": self.max_dist,
                "run_dir": str(paths.run_dir),
                "stage1_dir": str(paths.stage1_dir),
            },
            "pairs": matched_pairs,
            "id_map": id_map,
            "rgb_panels": rgb_panels,
            "ir_panels": ir_panels,
        }

        self._write_json(Path(paths.match_json_path), match_info)

        crop_pairs = self._collect_crop_pairs(
            rgb_crop_dir=Path(paths.rgb_crop_dir),
            ir_crop_dir=Path(paths.ir_crop_dir),
            matched_pairs=matched_pairs,
            id_map=id_map,
            rgb_panels=rgb_panels,
            ir_panels=ir_panels,
        )

        return Stage1Result(
            scene_name=scene_name,
            meta=match_info["meta"],
            rgb_panels=rgb_panels,
            ir_panels=ir_panels,
            pairs=matched_pairs,
            id_map={str(k): v for k, v in id_map.items()},
            crop_pairs=crop_pairs,
            paths=paths,
        )

    def _load_rgb_model(self) -> YOLO:
        if self._rgb_model is None:
            if not self.rgb_model_path.is_file():
                raise FileNotFoundError(f"找不到 RGB 模型權重：{self.rgb_model_path}")
            self._rgb_model = YOLO(str(self.rgb_model_path))
        return self._rgb_model

    def _load_ir_model(self) -> YOLO:
        if self._ir_model is None:
            if not self.ir_model_path.is_file():
                raise FileNotFoundError(f"找不到 IR 模型權重：{self.ir_model_path}")
            self._ir_model = YOLO(str(self.ir_model_path))
        return self._ir_model

    def _prepare_run_dirs(
        self,
        rgb_input_path: Path,
        ir_input_path: Path,
        scene_name: str,
        run_name: Optional[str],
        copy_inputs: bool,
    ) -> Stage1Paths:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_scene_name = self._sanitize_name(scene_name)

        if run_name is None:
            run_name = f"{timestamp}_{safe_scene_name}"

        run_dir = config.RUNS_DIR / run_name
        original_dir = run_dir / config.ORIGINAL_SUBDIR
        stage1_dir = run_dir / config.STAGE1_SUBDIR
        json_dir = run_dir / config.JSON_SUBDIR
        match_json_path = json_dir / config.MATCH_JSON_NAME

        for folder in [
            config.RUNS_DIR,
            config.CACHE_DIR,
            run_dir,
            original_dir,
            stage1_dir,
            json_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

        if copy_inputs:
            rgb_dst = original_dir / f"rgb{rgb_input_path.suffix.lower()}"
            ir_dst = original_dir / f"ir{ir_input_path.suffix.lower()}"
            shutil.copy2(rgb_input_path, rgb_dst)
            shutil.copy2(ir_input_path, ir_dst)
            final_rgb_path = rgb_dst
            final_ir_path = ir_dst
        else:
            final_rgb_path = rgb_input_path
            final_ir_path = ir_input_path

        rgb_crop_dir = stage1_dir / "RGB"
        ir_crop_dir = stage1_dir / "IR"

        return Stage1Paths(
            run_dir=str(run_dir),
            original_dir=str(original_dir),
            stage1_dir=str(stage1_dir),
            rgb_crop_dir=str(rgb_crop_dir),
            ir_crop_dir=str(ir_crop_dir),
            json_dir=str(json_dir),
            match_json_path=str(match_json_path),
            rgb_input_path=str(final_rgb_path),
            ir_input_path=str(final_ir_path),
        )

    def _collect_crop_pairs(
        self,
        rgb_crop_dir: Path,
        ir_crop_dir: Path,
        matched_pairs: List[Dict[str, Any]],
        id_map: Dict[Any, str],
        rgb_panels: List[Dict[str, Any]],
        ir_panels: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        crop_pairs: List[Dict[str, Any]] = []

        for pair in matched_pairs:
            rgb_idx = pair.get("rgb_idx")
            ir_idx = pair.get("ir_idx")

            panel_id = id_map.get(str(rgb_idx), id_map.get(rgb_idx))
            if panel_id is None:
                panel_id = f"idx{rgb_idx}"

            rgb_img = self._find_image_by_stem(rgb_crop_dir, panel_id)
            ir_img = self._find_image_by_stem(ir_crop_dir, panel_id)

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
                    "rgb_crop_path": str(rgb_img) if rgb_img else None,
                    "ir_crop_path": str(ir_img) if ir_img else None,
                    "rgb_xyxy": rgb_xyxy,
                    "ir_xyxy": ir_xyxy,
                }
            )

        crop_pairs.sort(key=lambda x: x["panel_id"])
        return crop_pairs

    def _find_image_by_stem(self, folder: Path, stem: str) -> Optional[Path]:
        if not folder.exists():
            return None
        for suffix in config.ALLOWED_IMAGE_SUFFIXES:
            candidate = folder / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _validate_input_image(self, path: Path, modality_name: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{modality_name} 影像不存在：{path}")
        if path.suffix.lower() not in config.ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(
                f"{modality_name} 影像副檔名不支援：{path.suffix}，"
                f"支援格式：{sorted(config.ALLOWED_IMAGE_SUFFIXES)}"
            )

    def _sanitize_name(self, name: str) -> str:
        valid_chars = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_"):
                valid_chars.append(ch)
            else:
                valid_chars.append("_")
        return "".join(valid_chars)


def run_stage1(
    rgb_input_path: str | Path,
    ir_input_path: str | Path,
    scene_name: Optional[str] = None,
    run_name: Optional[str] = None,
    copy_inputs: bool = True,
) -> Dict[str, Any]:
    pipeline = Stage1Pipeline()
    result = pipeline.run(
        rgb_input_path=rgb_input_path,
        ir_input_path=ir_input_path,
        scene_name=scene_name,
        run_name=run_name,
        copy_inputs=copy_inputs,
    )
    return result.to_dict()


if __name__ == "__main__":
    sample_rgb = config.TEMP_UPLOAD_DIR / "sample_1-1" / "rgb.jpg"
    sample_ir = config.TEMP_UPLOAD_DIR / "sample_1-1" / "ir.jpg"

    result = run_stage1(
        rgb_input_path=sample_rgb,
        ir_input_path=sample_ir,
        scene_name="sample_1-1",
    )

    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))
    print(f"match.json -> {result['paths']['match_json_path']}")