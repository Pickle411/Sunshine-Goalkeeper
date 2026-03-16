from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates


def render_page_header(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption("上傳一組 RGB / IR 影像，進行模組配對與瑕疵檢測展示。")


def render_upload_section() -> tuple:
    st.subheader("1. 上傳影像")

    col1, col2 = st.columns(2)
    with col1:
        rgb_file = st.file_uploader(
            "上傳 RGB 影像",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="rgb_uploader",
        )
    with col2:
        ir_file = st.file_uploader(
            "上傳 IR 影像",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="ir_uploader",
        )

    start = st.button("開始檢測", type="primary", use_container_width=True)
    return rgb_file, ir_file, start


def render_upload_preview(rgb_path: Optional[str], ir_path: Optional[str]) -> None:
    if not rgb_path and not ir_path:
        return

    st.subheader("2. 上傳預覽")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**RGB**")
        if rgb_path and Path(rgb_path).is_file():
            st.image(rgb_path, use_container_width=True)
    with col2:
        st.markdown("**IR**")
        if ir_path and Path(ir_path).is_file():
            st.image(ir_path, use_container_width=True)


def render_summary(summary: Dict[str, Any]) -> None:
    st.subheader("3. 結果摘要")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總模組數", summary.get("total_panels", 0))
    col2.metric("瑕疵模組數", summary.get("defective_panels", 0))
    col3.metric("正常模組數", summary.get("normal_panels", 0))
    col4.metric("總瑕疵數", summary.get("total_detections", 0))

    defect_type_count = summary.get("defect_type_count", {})
    if defect_type_count:
        st.markdown("**瑕疵類別統計**")
        for k, v in defect_type_count.items():
            st.write(f"- {k}: {v}")
    else:
        st.write("目前沒有偵測到瑕疵。")


def find_clicked_panel(
    panels: List[Dict[str, Any]],
    click_x: float,
    click_y: float,
) -> Optional[str]:
    for p in panels:
        bbox = p.get("rgb_xyxy")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            return p.get("panel_id")
    return None


def render_overview_section(
    overview_image_path: str,
    panels: List[Dict[str, Any]],
    current_selected_panel_id: Optional[str] = None,
) -> Optional[str]:
    st.subheader("4. 整區結果圖")
    st.caption("直接點擊紅色或綠色模組區塊，即可查看該模組細節。")

    if not overview_image_path or not Path(overview_image_path).is_file():
        st.warning("找不到總覽圖。")
        return current_selected_panel_id

    img = Image.open(overview_image_path)
    orig_w, orig_h = img.size

    # 顯示寬度可自行調整
    display_width = min(orig_w, 1000)
    scale_x = orig_w / display_width
    scale_y = orig_h / (orig_h * (display_width / orig_w))

    clicked = streamlit_image_coordinates(
        img,
        width=display_width,
        key="overview_click_map",
    )

    selected_panel_id = current_selected_panel_id

    if clicked is not None and "x" in clicked and "y" in clicked:
        click_x_display = clicked["x"]
        click_y_display = clicked["y"]

        # 依顯示比例換回原圖座標
        click_x_orig = click_x_display * scale_x
        click_y_orig = click_y_display * scale_y

        matched_panel_id = find_clicked_panel(
            panels=panels,
            click_x=click_x_orig,
            click_y=click_y_orig,
        )

        if matched_panel_id is not None:
            selected_panel_id = matched_panel_id

    if selected_panel_id:
        st.info(f"目前選擇模組：{selected_panel_id}")

    return selected_panel_id


def _split_detections_by_modality(detections: List[Dict[str, Any]]):
    """
    類別 0,1 -> IR
    類別 2,3 -> RGB
    """
    ir_dets = []
    rgb_dets = []

    for det in detections:
        cls_id = det.get("class_id", -1)
        if cls_id in {0, 1}:
            ir_dets.append(det)
        elif cls_id in {2, 3}:
            rgb_dets.append(det)

    return ir_dets, rgb_dets


def render_panel_detail(panel: Dict[str, Any]) -> None:
    st.subheader("5. 模組細節")

    detections = panel.get("detections_panel", [])
    ir_dets, rgb_dets = _split_detections_by_modality(detections)

    st.markdown("**結果顯示**")
    outer1, outer2 = st.columns([1, 1])

    with outer1:
        st.markdown("**RGB 結果圖**")
        left, center, right = st.columns([1.2, 1.6, 1.2])
        with center:
            rgb_vis_path = panel.get("vis_path")
            if rgb_vis_path and Path(rgb_vis_path).is_file():
                st.image(rgb_vis_path, use_container_width=True)
            else:
                st.info("找不到 RGB 結果圖。")

    with outer2:
        st.markdown("**IR 結果圖**")
        left, center, right = st.columns([1.2, 1.6, 1.2])
        with center:
            ir_vis_path = panel.get("ir_vis_path")
            if ir_vis_path and Path(ir_vis_path).is_file():
                st.image(ir_vis_path, use_container_width=True)
            else:
                st.info("找不到 IR 結果圖。")

    st.divider()

    info1, info2 = st.columns([1, 1])

    with info1:
        st.markdown("**基本資訊**")
        st.write(f"Panel ID: {panel.get('panel_id')}")
        st.write(f"狀態: {panel.get('status')}")
        st.write(f"是否瑕疵: {'是' if panel.get('is_defective') else '否'}")
        st.write(f"瑕疵數量: {panel.get('num_defects', 0)}")
        st.write(f"RGB 座標: {panel.get('rgb_xyxy')}")

    with info2:
        st.markdown("**模態對應說明**")
        st.write("IR 類別: 0, 1")
        st.write("RGB 類別: 2, 3")

    det_col1, det_col2 = st.columns([1, 1])

    with det_col1:
        st.markdown("**RGB 瑕疵明細 (類別 2,3)**")
        if rgb_dets:
            for i, det in enumerate(rgb_dets, start=1):
                st.write(
                    f"{i}. 類別: {det.get('class_name')} | "
                    f"信心分數: {det.get('confidence')} | "
                    f"bbox: {det.get('bbox_xyxy')}"
                )
        else:
            st.write("此模組沒有 RGB 類瑕疵。")

    with det_col2:
        st.markdown("**IR 瑕疵明細 (類別 0,1)**")
        if ir_dets:
            for i, det in enumerate(ir_dets, start=1):
                st.write(
                    f"{i}. 類別: {det.get('class_name')} | "
                    f"信心分數: {det.get('confidence')} | "
                    f"bbox: {det.get('bbox_xyxy')}"
                )
        else:
            st.write("此模組沒有 IR 類瑕疵。")


def render_run_info(run_dir: str, raw_paths: Dict[str, Any]) -> None:
    with st.expander("顯示執行資訊"):
        st.write(f"Run 目錄: {run_dir}")
        for k, v in raw_paths.items():
            st.write(f"{k}: {v}")


def render_error(message: str) -> None:
    st.error(message)


def render_success(message: str) -> None:
    st.success(message)
