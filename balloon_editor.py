"""
吹き出し配置エディタ - Streamlit UI

縦並び漫画画像に吹き出しを配置するUIです。

使用方法:
    streamlit run balloon_editor.py
"""

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import numpy as np
import json
from pathlib import Path
import io
import base64
import textwrap


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


def composite_balloons(base_image: Image.Image, placements: list, balloons: list) -> Image.Image:
    """吹き出しを配置した画像を合成"""
    result = base_image.copy()
    
    for placement in placements:
        balloon_idx = placement["balloon_idx"]
        if balloon_idx >= len(balloons):
            continue
        balloon_img = balloons[balloon_idx]["image"]
        x = placement["x"]
        y = placement["y"]
        scale = placement.get("scale", 1.0)
        
        # スケーリング
        if scale != 1.0:
            new_w = int(balloon_img.width * scale)
            new_h = int(balloon_img.height * scale)
            if new_w > 0 and new_h > 0:
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
                    
                    st.image(thumb, caption=f"#{i}", width='stretch')
            
            st.divider()
            
            # 選択中の吹き出し
            if st.session_state.selected_balloon is not None:
                idx = st.session_state.selected_balloon
                st.success(f"選択中: #{idx}")
                st.image(balloons[idx]["image"], width='stretch')
                
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
    
    # 右側固定パネル用のCSS
    st.markdown("""
    <style>
    /* メインコンテンツエリア */
    .main .block-container {
        padding-right: 380px;
        max-width: 100%;
    }
    
    /* 右側固定パネル */
    .right-panel {
        position: fixed;
        right: 1rem;
        top: 4rem;
        width: 350px;
        height: calc(100vh - 5rem);
        overflow-y: auto;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        z-index: 100;
        box-shadow: -2px 0 10px rgba(0,0,0,0.1);
    }
    
    .right-panel h3 {
        margin-top: 0;
        color: #333;
        font-size: 1.2rem;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0.5rem;
    }
    
    .right-panel img {
        width: 100%;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    /* メイン画像の幅を制限 */
    .main-image-container {
        max-width: 500px;
    }
    
    /* ボタンのスタイル調整 */
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 右側固定パネルの内容を準備（HTMLで直接レンダリング）
    # インデント無しの単一行HTMLで定義
    panel_html = '<div class="right-panel"><h3>📖 操作ガイド</h3>'
    
    if st.session_state.selected_balloon is not None:
        panel_html += f'<p style="background-color: #d1ecf1; padding: 10px; border-radius: 5px; color: #0c5460;">🎈 吹き出し #{st.session_state.selected_balloon} を選択中<br>👈 左の縦並び画像をクリックして配置</p>'
    else:
        panel_html += '<p style="background-color: #d1ecf1; padding: 10px; border-radius: 5px; color: #0c5460;">👈 左のサイドバーから吹き出しを選択し、縦並び画像をクリックして配置</p>'
    
    # オリジナル画像を表示
    if original_image:
        # 画像をbase64エンコード
        buffered = io.BytesIO()
        original_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        panel_html += f'<h3>🖼️ オリジナル画像</h3><img src="data:image/png;base64,{img_str}" alt="Original Image">'
    
    panel_html += '</div>'
    
    # 右側パネルを表示
    st.markdown(panel_html, unsafe_allow_html=True)
    
    # 保存ボタン
    st.subheader("💾 保存")
    col_save1, col_save2, col_save3 = st.columns(3)
    
    with col_save1:
        if st.button("💾 画像を保存", type="primary", use_container_width=True):
            # 最終画像を生成
            final_image = composite_balloons(vertical_image, st.session_state.placements, balloons)
            
            # PNG形式で保存
            save_path = folder_path / "vertical_with_balloons.png"
            final_image.save(save_path)
            
            st.success(f"✅ 保存しました: {save_path}")
    
    with col_save2:
        if st.button("📄 配置情報を保存", use_container_width=True):
            # 配置情報をJSONで保存
            save_data = {
                "folder": selected_folder,
                "placements": st.session_state.placements
            }
            
            json_path = folder_path / "balloon_placements.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ 保存しました: {json_path}")
    
    with col_save3:
        # 配置情報の読み込み
        json_path = folder_path / "balloon_placements.json"
        if json_path.exists():
            if st.button("📂 配置情報を読み込み", use_container_width=True):
                with open(json_path, "r", encoding="utf-8") as f:
                    save_data = json.load(f)
                st.session_state.placements = save_data.get("placements", [])
                st.success("✅ 読み込みました")
                st.rerun()
    
    st.divider()
    
    # メインエリア：縦並び画像表示
    st.subheader("縦並び画像（クリックで吹き出しを配置）")
    
    # 現在の配置を反映した画像を作成
    preview_image = composite_balloons(vertical_image, st.session_state.placements, balloons)
    
    # 画像をクリック可能にして表示（幅を500pxに固定）
    display_width = 500
    scale_factor = display_width / preview_image.width
    
    # クリック可能な画像表示
    st.markdown('<div class="main-image-container">', unsafe_allow_html=True)
    coords = streamlit_image_coordinates(
        preview_image,
        key=f"clickable_image_{st.session_state.current_folder}_{len(st.session_state.placements)}",
        width=display_width
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if coords is not None and st.session_state.selected_balloon is not None:
        # クリック座標を元の画像座標に変換
        click_x = int(coords["x"] / scale_factor)
        click_y = int(coords["y"] / scale_factor)
        
        # 新しい配置を追加
        new_placement = {
            "balloon_idx": st.session_state.selected_balloon,
            "x": click_x,
            "y": click_y,
            "scale": scale
        }
        st.session_state.placements.append(new_placement)
        st.success(f"✅ 配置しました: ({click_x}, {click_y})")
        st.rerun()


if __name__ == "__main__":
    main()