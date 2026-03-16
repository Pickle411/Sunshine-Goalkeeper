from datetime import datetime
from pathlib import Path

import streamlit as st

import config
from ui_components import (
    render_error,
    render_overview_section,
    render_page_header,
    render_panel_detail,
    render_run_info,
    render_success,
    render_summary,
    render_upload_preview,
    render_upload_section,
)


def save_uploaded_file(uploaded_file, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def prepare_temp_input_paths(rgb_file, ir_file) -> tuple[Path, Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_name = f"upload_{timestamp}"
    sample_dir = config.TEMP_UPLOAD_DIR / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb_ext = Path(rgb_file.name).suffix.lower() or ".jpg"
    ir_ext = Path(ir_file.name).suffix.lower() or ".jpg"

    rgb_path = sample_dir / f"rgb{rgb_ext}"
    ir_path = sample_dir / f"ir{ir_ext}"

    save_uploaded_file(rgb_file, rgb_path)
    save_uploaded_file(ir_file, ir_path)

    return rgb_path, ir_path, sample_name


def run_full_pipeline(rgb_path: Path, ir_path: Path, scene_name: str) -> dict:
    # lazy import，避免首頁黑畫面
    from services.stage1_pipeline import run_stage1
    from services.stage2_inference import run_stage2_from_stage1_result
    from services.overview_renderer import run_overview_renderer
    from services.result_formatter import build_ui_payload

    stage1_result = run_stage1(
        rgb_input_path=rgb_path,
        ir_input_path=ir_path,
        scene_name=scene_name,
    )

    run_dir = stage1_result["paths"]["run_dir"]

    run_stage2_from_stage1_result(stage1_result)
    run_overview_renderer(run_dir)
    payload = build_ui_payload(run_dir)

    payload["run_dir"] = run_dir
    payload["input_rgb_path"] = str(rgb_path)
    payload["input_ir_path"] = str(ir_path)
    return payload


def main():
    render_page_header(config.APP_TITLE)

    rgb_file, ir_file, start = render_upload_section()

    if "ui_payload" not in st.session_state:
        st.session_state.ui_payload = None

    if "selected_panel_id" not in st.session_state:
        st.session_state.selected_panel_id = None

    if rgb_file and ir_file:
        rgb_path_preview, ir_path_preview, _ = prepare_temp_input_paths(rgb_file, ir_file)
        render_upload_preview(str(rgb_path_preview), str(ir_path_preview))

    if start:
        if not rgb_file or not ir_file:
            render_error("請先上傳 RGB 與 IR 影像。")
        else:
            try:
                rgb_path, ir_path, scene_name = prepare_temp_input_paths(rgb_file, ir_file)

                with st.spinner("正在執行完整流程..."):
                    payload = run_full_pipeline(rgb_path, ir_path, scene_name)

                st.session_state.ui_payload = payload
                st.session_state.selected_panel_id = None
                render_success("檢測完成。")

            except Exception as e:
                render_error(f"執行失敗: {e}")

    payload = st.session_state.ui_payload
    if payload:
        render_summary(payload["summary"])

        left_col, right_col = st.columns([1.35, 1])

        with left_col:
            selected_panel_id = render_overview_section(
                payload["overview"]["image_path"],
                payload["panels"],
                current_selected_panel_id=st.session_state.selected_panel_id,
            )

        st.session_state.selected_panel_id = selected_panel_id

        with right_col:
            # st.subheader("5. 模組細節")
            if selected_panel_id:
                selected_panel = next(
                    (p for p in payload["panels"] if p["panel_id"] == selected_panel_id),
                    None,
                )
                if selected_panel:
                    render_panel_detail(selected_panel)
            else:
                st.info("請先點擊左側整區結果圖中的模組。")

        render_run_info(payload["run_dir"], payload["raw_paths"])


if __name__ == "__main__":
    main()
