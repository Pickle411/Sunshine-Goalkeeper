from pathlib import Path

# =========================
# 專案根目錄
# =========================
UI_ROOT = Path("/mnt/shared/chuanyu_m11317028/SolarPanel/UI")
PROJECT_ROOT = Path("/mnt/shared/chuanyu_m11317028/SolarPanel/YOLOv11-RGBT")

# 你原本 detect_panels / match_panels / crop_panels 的實際位置
PANEL_TOOL_ROOT = PROJECT_ROOT / "DataPreprocessingTools" / "SolarPanelModule"

# =========================
# 資料夾
# =========================
TEMP_UPLOAD_DIR = UI_ROOT / "data" / "temp_uploads"
RUNS_DIR = UI_ROOT / "data" / "runs"
CACHE_DIR = UI_ROOT / "data" / "cache"

# =========================
# 模型權重
# =========================
MODULE_IR_MODEL_PATH = UI_ROOT / "pt_file" / "Module" / "IR_best.pt"
MODULE_RGB_MODEL_PATH = UI_ROOT / "pt_file" / "Module" / "RGB_best.pt"
ANOMALY_MODEL_PATH = UI_ROOT / "pt_file" / "Anormaly" / "best.pt"

# =========================
# 裝置
# =========================
DEVICE = "cuda:0"

# =========================
# Stage 1 參數
# 沿用你原本 1.0-run_pipeline.py
# =========================
MODULE_CONF_THRES = 0.5
ROW_EPS = 0.03
MAX_DIST = 0.05

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# =========================
# UI 顯示
# =========================
APP_TITLE = "Solar Panel Defect Detection Demo"

# =========================
# 輸出資料夾命名
# =========================
ORIGINAL_SUBDIR = "original"
STAGE1_SUBDIR = "stage1"
RGB_CROP_SUBDIR = "RGB"
IR_CROP_SUBDIR = "IR"
JSON_SUBDIR = "json"
MATCH_JSON_NAME = "match.json"

# =========================
# 是否保留中間輸出
# =========================
KEEP_STAGE1_CROPS = True