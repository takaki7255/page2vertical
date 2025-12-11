"""
吹き出し配置エディタ - Streamlit UI

縦並び漫画画像に吹き出しをドラッグ&ドロップで配置するUIです。

使用方法:
    streamlit run balloon_editor.py
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import json
from pathlib import Path
import cv2
import os


def load_output_folder(folder_path: str):
    """出力フォルダから画像データを読み込む"""
    folder = Path(folder_path)
    
    # 縦並び画像
    vertical_path = folder / "vertical.png"
    if not vertical_path.exists():
        return None, None, None, None
    
    vertical_image = Image.open(vertical_path).convert("RGBA")
    
    # オリジナル画像
    original_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        p = folder / f"original{ext}"
        if p.exists():
            original_path = p
            break
    
    original_image = None
    if original_path:
        original_image = Image.open(original_path).convert("RGBA")
    
    # 吹き出し画像
    balloons_dir = folder / "balloons"
    balloons = []
    if balloons_dir.exists():
        for balloon_path in sorted(balloons_dir.glob("balloon_*.png")):
            balloon_img = Image.open(balloon_path).convert("RGBA")
            balloons.append({
                "path": str(balloon_path),
                "name": balloon_path.name,
                "image": balloon_img
            })
    
    # メタ情報
    meta_path = folder / "meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    
    return vertical_image, original_image, balloons, meta


def get_output_folders(base_dir: str):
    """出力ディレクトリ内のフォルダ一覧を取得"""
    base = Path(base_dir)
    if not base.exists():
        return []
    
    folders = []
    for item in sorted(base.iterdir()):
        if item.is_dir() and (item / "vertical.png").exists():
            folders.append(item.name)
    
    return folders


def composite_balloons(base_image: Image.Image, placements: list) -> Image.Image:
    """吹き出しを配置した画像を合成"""
    result = base_image.copy()
    
    for placement in placements:
        balloon_img = placement["image"]
        x = placement["x"]
        y = placement["y"]
        scale = placement.get("scale", 1.0)
        
        # スケーリング
        if scale != 1.0:
            new_w = int(balloon_img.width * scale)
            new_h = int(balloon_img.height * scale)
            balloon_img = balloon_img.resize((new_w, new_h), Image.LANCZOS)
        
        # 配置（中心基準）
        paste_x = int(x - balloon_img.width / 2)
        paste_y = int(y - balloon_img.height / 2)
        
        # 画像範囲内にクリップ
        paste_x = max(0, min(paste_x, result.width - balloon_img.width))
        paste_y = max(0, min(paste_y, result.height - balloon_img.height))
        
        # アルファ合成
        result.paste(balloon_img, (paste_x, paste_y), balloon_img)
    
    return result


def main():
    st.set_page_config(
        page_title="吹き出し配置エディタ",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 吹き出し配置エディタ")
    
    # サイドバー：フォルダ選択
    with st.sidebar:
        st.header("📁 フォルダ選択")
        
        output_base = st.text_input(
            "出力ベースディレクトリ",
            value="./output_m2f",
            help="main.pyの出力先ディレクトリを指定"
        )
        
        folders = get_output_folders(output_base)
        
        if not folders:
            st.warning("出力フォルダが見つかりません")
            return
        
        selected_folder = st.selectbox(
            "画像フォルダを選択",
            folders
        )
        
        folder_path = Path(output_base) / selected_folder
        
        st.divider()
        st.header("🎈 吹き出し")
    
    # データ読み込み
    vertical_image, original_image, balloons, meta = load_output_folder(folder_path)
    
    if vertical_image is None:
        st.error("縦並び画像が見つかりません")
        return
    
    # セッション状態の初期化
    if "placements" not in st.session_state:
        st.session_state.placements = []
    
    if "selected_balloon" not in st.session_state:
        st.session_state.selected_balloon = None
    
    if "current_folder" not in st.session_state:
        st.session_state.current_folder = None
    
    # フォルダが変わったらリセット
    if st.session_state.current_folder != selected_folder:
        st.session_state.placements = []
        st.session_state.selected_balloon = None
        st.session_state.current_folder = selected_folder
    
    # サイドバー：吹き出し選択
    with st.sidebar:
        if balloons:
            # 吹き出しをグリッド表示
            cols = st.columns(2)
            for i, balloon in enumerate(balloons):
                with cols[i % 2]:
                    # サムネイル表示
                    thumb = balloon["image"].copy()
                    thumb.thumbnail((100, 100))
                    
                    if st.button(
                        f"#{i}",
                        key=f"balloon_{i}",
                        help=balloon["name"]
                    ):
                        st.session_state.selected_balloon = i
                    
                    st.image(thumb, caption=f"#{i}", use_container_width=True)
            
            st.divider()
            
            # 選択中の吹き出し
            if st.session_state.selected_balloon is not None:
                idx = st.session_state.selected_balloon
                st.success(f"選択中: #{idx}")
                st.image(balloons[idx]["image"], use_container_width=True)
                
                # スケール調整
                scale = st.slider("サイズ", 0.5, 2.0, 1.0, 0.1, key="balloon_scale")
            else:
                st.info("吹き出しを選択してください")
                scale = 1.0
        else:
            st.warning("吹き出しがありません")
            scale = 1.0
        
        st.divider()
        
        # 配置済みリスト
        st.header("📍 配置済み")
        if st.session_state.placements:
            for i, p in enumerate(st.session_state.placements):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"#{p['balloon_idx']} ({p['x']:.0f}, {p['y']:.0f})")
                with col2:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.placements.pop(i)
                        st.rerun()
        else:
            st.text("なし")
        
        if st.button("すべてクリア", type="secondary"):
            st.session_state.placements = []
            st.rerun()
    
    # メインエリア：画像表示
    col_main, col_orig = st.columns([2, 1])
    
    with col_main:
        st.subheader("縦並び画像（クリックで配置）")
        
        # 画像サイズを調整（表示用）
        display_height = 800
        aspect_ratio = vertical_image.width / vertical_image.height
        display_width = int(display_height * aspect_ratio)
        
        # 現在の配置を反映した画像を作成
        placements_with_images = []
        for p in st.session_state.placements:
            balloon_img = balloons[p["balloon_idx"]]["image"]
            placements_with_images.append({
                "image": balloon_img,
                "x": p["x"],
                "y": p["y"],
                "scale": p.get("scale", 1.0)
            })
        
        preview_image = composite_balloons(vertical_image, placements_with_images)
        
        # 表示用にリサイズ
        preview_resized = preview_image.resize((display_width, display_height), Image.LANCZOS)
        
        # Canvas（クリック検出用）
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=preview_resized,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="point",
            point_display_radius=5,
            key="canvas",
        )
        
        # クリック位置を取得
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            
            if objects and st.session_state.selected_balloon is not None:
                # 最新のクリック位置を取得
                last_obj = objects[-1]
                click_x = last_obj.get("left", 0)
                click_y = last_obj.get("top", 0)
                
                # 表示サイズから元サイズへの変換
                scale_x = vertical_image.width / display_width
                scale_y = vertical_image.height / display_height
                
                real_x = click_x * scale_x
                real_y = click_y * scale_y
                
                # 新しい配置を追加
                new_placement = {
                    "balloon_idx": st.session_state.selected_balloon,
                    "x": real_x,
                    "y": real_y,
                    "scale": scale
                }
                
                # 重複チェック（同じ位置への配置を防ぐ）
                is_duplicate = False
                for p in st.session_state.placements:
                    if (abs(p["x"] - real_x) < 10 and 
                        abs(p["y"] - real_y) < 10 and
                        p["balloon_idx"] == st.session_state.selected_balloon):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    st.session_state.placements.append(new_placement)
                    st.rerun()
    
    with col_orig:
        st.subheader("オリジナル画像（参照用）")
        if original_image:
            st.image(original_image, use_container_width=True)
        else:
            st.warning("オリジナル画像がありません")
    
    # 保存ボタン
    st.divider()
    col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
    
    with col_save1:
        if st.button("💾 画像を保存", type="primary"):
            # 最終画像を生成
            final_image = composite_balloons(vertical_image, placements_with_images)
            
            # PNG形式で保存
            save_path = folder_path / "vertical_with_balloons.png"
            final_image.save(save_path)
            
            st.success(f"保存しました: {save_path}")
    
    with col_save2:
        if st.button("📄 配置情報を保存"):
            # 配置情報をJSONで保存
            save_data = {
                "folder": selected_folder,
                "placements": st.session_state.placements
            }
            
            json_path = folder_path / "balloon_placements.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"保存しました: {json_path}")
    
    with col_save3:
        # 配置情報の読み込み
        json_path = folder_path / "balloon_placements.json"
        if json_path.exists():
            if st.button("📂 配置情報を読み込み"):
                with open(json_path, "r", encoding="utf-8") as f:
                    save_data = json.load(f)
                st.session_state.placements = save_data.get("placements", [])
                st.rerun()


if __name__ == "__main__":
    main()
