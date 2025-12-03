"""
見開きページ漫画を縦読みに変換するツール

使用方法:
    python main.py --input image.jpg --output ./output/
    python main.py --input ./manga_pages/ --output ./output/
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
import json
import os

# predict.pyからインポート
from predict import (
    create_maskrcnn,
    create_mask2former,
    load_image,
    predict_maskrcnn,
    predict_mask2former,
)

# panel_order_estimater.pyからインポート
from panel_order import panel_order_estimater

# Lamaインペイント用
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    print("Warning: simple_lama_inpainting is not installed. Inpainting will be skipped.")


# ================== 設定 ==================
DEFAULT_PANEL_MODEL_PATH = "./instance_models/maskrcnn_gray_20251201_121013/maskrcnn_gray_best.pt"
DEFAULT_PANEL_MODEL_TYPE = "maskrcnn"
DEFAULT_INPUT_TYPE = "3ch"
DEFAULT_IMG_SIZE = (384, 512)
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_PANEL_MARGIN = 20  # コマ間の余白（ピクセル）


class BalloonSegmenter:
    """
    吹き出しのセマンティックセグメンテーション
    U-Netベースのモデルを使用（将来実装用のプレースホルダー）
    """
    def __init__(self, model_path: Optional[str] = None, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """モデルを読み込む（将来実装）"""
        # TODO: 吹き出しセグメンテーションモデルの読み込み
        pass
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        吹き出し領域をセグメンテーション
        
        Args:
            image: BGR画像 (H, W, 3)
        
        Returns:
            mask: バイナリマスク (H, W) 吹き出し領域が255
        """
        if self.model is None:
            # モデルがない場合は簡易的な吹き出し検出（白色領域検出）
            return self._simple_balloon_detection(image)
        
        # TODO: 学習済みモデルによるセグメンテーション
        return self._simple_balloon_detection(image)
    
    def _simple_balloon_detection(self, image: np.ndarray) -> np.ndarray:
        """
        簡易的な吹き出し検出（白色領域をベースに検出）
        実際の使用時はU-Net等の学習済みモデルに置き換える
        
        Args:
            image: BGR画像 (H, W, 3)
        
        Returns:
            mask: バイナリマスク (H, W)
        """
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二値化（白色領域を検出）
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # モルフォロジー処理でノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出して小さい領域を除去
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary)
        
        min_area = 500  # 最小面積閾値
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 円形度をチェック（吹き出しは比較的丸い）
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.1:  # ある程度丸い形状
                        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask


class LamaInpainter:
    """Lamaによるインペイント処理"""
    
    def __init__(self):
        if LAMA_AVAILABLE:
            self.lama = SimpleLama()
        else:
            self.lama = None
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, dilate_iterations: int = 3) -> np.ndarray:
        """
        マスク領域をインペイント
        
        Args:
            image: BGR画像 (H, W, 3)
            mask: バイナリマスク (H, W) インペイント領域が255
            dilate_iterations: マスク膨張回数
        
        Returns:
            inpainted_image: インペイント済み画像 (H, W, 3)
        """
        if self.lama is None:
            # Lamaが利用できない場合はOpenCVのインペイントを使用
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # PIL形式に変換
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # マスクを膨張
        from scipy.ndimage import binary_dilation
        binary_mask = mask > 127
        dilated_mask = binary_dilation(binary_mask, iterations=dilate_iterations)
        mask_array = (dilated_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_array)
        
        # インペイント実行
        result = self.lama(image_pil, mask_pil)
        
        # BGR形式に戻す
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


class MangaPage2Vertical:
    """見開き漫画ページを縦読み形式に変換"""
    
    def __init__(
        self,
        panel_model_path: str = DEFAULT_PANEL_MODEL_PATH,
        panel_model_type: str = DEFAULT_PANEL_MODEL_TYPE,
        balloon_model_path: Optional[str] = None,
        input_type: str = DEFAULT_INPUT_TYPE,
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        panel_margin: int = DEFAULT_PANEL_MARGIN,
        device: Optional[torch.device] = None
    ):
        # デバイス設定
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        print(f"Device: {self.device}")
        
        # パラメータ保存
        self.panel_model_type = panel_model_type
        self.input_type = input_type
        self.img_size = img_size
        self.score_threshold = score_threshold
        self.panel_margin = panel_margin
        
        # コマ検出モデルの読み込み
        self._load_panel_model(panel_model_path, panel_model_type)
        
        # 吹き出しセグメンテーション
        self.balloon_segmenter = BalloonSegmenter(balloon_model_path, self.device)
        
        # インペインター
        self.inpainter = LamaInpainter()
    
    def _load_panel_model(self, model_path: str, model_type: str):
        """コマ検出モデルを読み込む"""
        print(f"Loading {model_type} model from {model_path}...")
        
        if model_type == 'maskrcnn':
            self.panel_model = create_maskrcnn(num_classes=2)
            self.model_name = None
        else:  # mask2former
            self.panel_model, self.model_name = create_mask2former(num_classes=2)
        
        # 重み読み込み
        state_dict = torch.load(model_path, map_location=self.device)
        self.panel_model.load_state_dict(state_dict)
        self.panel_model = self.panel_model.to(self.device)
        self.panel_model.eval()
        
        print("Model loaded successfully.")
    
    def is_spread_page(self, image: np.ndarray) -> bool:
        """見開きページかどうかを判定（横が縦より長い場合）"""
        h, w = image.shape[:2]
        return w > h
    
    def split_spread_page(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        見開きページを左右に分割
        
        Returns:
            (right_page, left_page): 右ページが先（読み順）
        """
        h, w = image.shape[:2]
        mid = w // 2
        
        right_page = image[:, mid:].copy()  # 右側（先に読む）
        left_page = image[:, :mid].copy()   # 左側（後に読む）
        
        return right_page, left_page
    
    def detect_panels(self, image: np.ndarray, image_path: Optional[str] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        コマのインスタンスセグメンテーション
        
        Args:
            image: BGR画像
            image_path: 画像パス（LSD/SDF生成に使用）
        
        Returns:
            masks: マスクのリスト
            panels_info: コマ情報のリスト（バウンディングボックス等）
        """
        orig_h, orig_w = image.shape[:2]
        
        # 一時ファイルとして保存（load_imageがファイルパスを必要とするため）
        if image_path is None:
            temp_path = "/tmp/temp_panel_detect.jpg"
            cv2.imwrite(temp_path, image)
            image_path = temp_path
        
        # 画像読み込みと前処理
        image_tensor, _, _ = load_image(
            image_path, self.img_size, self.input_type
        )
        
        # 予測
        if self.panel_model_type == 'maskrcnn':
            masks, scores, boxes = predict_maskrcnn(
                self.panel_model, image_tensor, self.device, self.score_threshold
            )
        else:
            masks, scores, boxes = predict_mask2former(
                self.panel_model, image_tensor, self.device, self.model_name
            )
        
        # マスクを元画像サイズにリサイズ
        resized_masks = []
        panels_info = []
        
        scale_x = orig_w / self.img_size[1]
        scale_y = orig_h / self.img_size[0]
        
        for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
            # マスクをリサイズ
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(mask_resized)
            
            # バウンディングボックスをスケール
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            panels_info.append({
                'type': 'frame',
                'id': f'{i:08d}',
                'xmin': str(x1),
                'ymin': str(y1),
                'xmax': str(x2),
                'ymax': str(y2),
                'score': float(score),
                'mask': mask_resized
            })
        
        return resized_masks, panels_info
    
    def estimate_panel_order(self, panels_info: List[Dict], img_width: int, img_height: int) -> List[Dict]:
        """コマの読み順を推定"""
        # panel_order_estimater用にデータを整形
        panels_for_order = []
        for panel in panels_info:
            panels_for_order.append({
                'type': panel['type'],
                'id': panel['id'],
                'xmin': panel['xmin'],
                'ymin': panel['ymin'],
                'xmax': panel['xmax'],
                'ymax': panel['ymax'],
                'score': panel.get('score', 1.0),
                'mask': panel.get('mask')
            })
        
        # 順序推定
        ordered_panels = panel_order_estimater(panels_for_order, img_width, img_height)
        
        return ordered_panels
    
    def extract_balloons(self, image: np.ndarray, balloon_mask: np.ndarray) -> List[Dict]:
        """
        吹き出しを個別に抽出
        
        Returns:
            balloons: 吹き出し情報のリスト
                [{'image': np.ndarray, 'mask': np.ndarray, 'bbox': (x, y, w, h)}, ...]
        """
        # 連結成分分析で個々の吹き出しを分離
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(balloon_mask)
        
        balloons = []
        for i in range(1, num_labels):  # 0は背景
            x, y, w, h, area = stats[i]
            
            if area < 100:  # 小さすぎるものは除外
                continue
            
            # 吹き出し領域を切り出し
            balloon_roi = image[y:y+h, x:x+w].copy()
            mask_roi = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
            
            # マスク領域のみを抽出（透過用）
            balloon_rgba = cv2.cvtColor(balloon_roi, cv2.COLOR_BGR2BGRA)
            balloon_rgba[:, :, 3] = mask_roi
            
            balloons.append({
                'image': balloon_rgba,
                'mask': mask_roi,
                'bbox': (x, y, w, h),
                'centroid': (centroids[i][0], centroids[i][1])
            })
        
        return balloons
    
    def extract_panel_image(self, image: np.ndarray, panel_info: Dict) -> np.ndarray:
        """
        コマ領域を切り出す
        
        Args:
            image: 元画像（BGR）
            panel_info: コマ情報（バウンディングボックスとマスク）
        
        Returns:
            panel_image: 切り出したコマ画像
        """
        xmin = int(panel_info['xmin'])
        ymin = int(panel_info['ymin'])
        xmax = int(panel_info['xmax'])
        ymax = int(panel_info['ymax'])
        
        # マスクがある場合はマスク領域のみを抽出
        if 'mask' in panel_info and panel_info['mask'] is not None:
            mask = panel_info['mask']
            
            # マスク領域の背景を白に
            panel_image = image.copy()
            panel_image[mask == 0] = [255, 255, 255]  # 白背景
            
            # バウンディングボックスで切り出し
            panel_image = panel_image[ymin:ymax, xmin:xmax]
        else:
            panel_image = image[ymin:ymax, xmin:xmax].copy()
        
        return panel_image
    
    def create_vertical_manga(
        self,
        panels: List[np.ndarray],
        margin: int = None,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        コマを縦に並べて縦読み漫画を作成
        
        Args:
            panels: コマ画像のリスト（読み順）
            margin: コマ間の余白
            background_color: 背景色（BGR）
        
        Returns:
            vertical_manga: 縦読み漫画画像
        """
        if margin is None:
            margin = self.panel_margin
        
        if not panels:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 最大幅を計算
        max_width = max(panel.shape[1] for panel in panels)
        
        # 総高さを計算
        total_height = sum(panel.shape[0] for panel in panels) + margin * (len(panels) + 1)
        
        # 出力画像を作成
        vertical_manga = np.full((total_height, max_width, 3), background_color, dtype=np.uint8)
        
        # コマを配置
        current_y = margin
        for panel in panels:
            h, w = panel.shape[:2]
            
            # 中央揃えで配置
            x_offset = (max_width - w) // 2
            
            vertical_manga[current_y:current_y+h, x_offset:x_offset+w] = panel
            current_y += h + margin
        
        return vertical_manga
    
    def process_page(self, image: np.ndarray, image_path: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        単一ページを処理
        
        Returns:
            (ordered_panels_images, balloons): 順序付けられたコマ画像リストと吹き出しリスト
        """
        h, w = image.shape[:2]
        
        # 1. 吹き出しセグメンテーション
        balloon_mask = self.balloon_segmenter.segment(image)
        
        # 2. 吹き出しを個別に抽出
        balloons = self.extract_balloons(image, balloon_mask)
        
        # 3. 吹き出し領域をインペイント
        inpainted_image = self.inpainter.inpaint(image, balloon_mask)
        
        # 4. コマのインスタンスセグメンテーション
        masks, panels_info = self.detect_panels(inpainted_image, image_path)
        
        if not panels_info:
            print("  No panels detected.")
            return [], balloons
        
        # 5. コマの順序推定
        ordered_panels = self.estimate_panel_order(panels_info, w, h)
        
        # 6. 順序通りにコマを切り出し
        ordered_panel_images = []
        for panel in ordered_panels:
            panel_image = self.extract_panel_image(inpainted_image, panel)
            ordered_panel_images.append(panel_image)
        
        return ordered_panel_images, balloons
    
    def convert(self, image_path: str, output_dir: str) -> Dict:
        """
        見開き漫画を縦読みに変換
        
        Args:
            image_path: 入力画像パス
            output_dir: 出力ディレクトリ
        
        Returns:
            result: 処理結果の情報
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Processing: {image_path.name}")
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
        
        all_panels = []
        all_balloons = []
        
        # 見開きページかチェック
        if self.is_spread_page(image):
            print("  Detected as spread page. Splitting...")
            
            # 右ページ（先）と左ページ（後）に分割
            right_page, left_page = self.split_spread_page(image)
            
            # 右ページを処理
            print("  Processing right page...")
            right_temp_path = "/tmp/temp_right_page.jpg"
            cv2.imwrite(right_temp_path, right_page)
            right_panels, right_balloons = self.process_page(right_page, right_temp_path)
            all_panels.extend(right_panels)
            all_balloons.extend(right_balloons)
            
            # 左ページを処理
            print("  Processing left page...")
            left_temp_path = "/tmp/temp_left_page.jpg"
            cv2.imwrite(left_temp_path, left_page)
            left_panels, left_balloons = self.process_page(left_page, left_temp_path)
            all_panels.extend(left_panels)
            all_balloons.extend(left_balloons)
        else:
            print("  Detected as single page.")
            panels, balloons = self.process_page(image, str(image_path))
            all_panels.extend(panels)
            all_balloons.extend(balloons)
        
        print(f"  Total panels: {len(all_panels)}")
        print(f"  Total balloons: {len(all_balloons)}")
        
        # 縦読み漫画を作成
        if all_panels:
            vertical_manga = self.create_vertical_manga(all_panels)
            
            # 結果を保存
            output_path = output_dir / f"{image_path.stem}_vertical.png"
            cv2.imwrite(str(output_path), vertical_manga)
            print(f"  Saved vertical manga: {output_path}")
        else:
            output_path = None
            print("  No panels to create vertical manga.")
        
        # 吹き出しを保存
        balloons_dir = output_dir / f"{image_path.stem}_balloons"
        if all_balloons:
            balloons_dir.mkdir(exist_ok=True)
            
            for i, balloon in enumerate(all_balloons):
                balloon_path = balloons_dir / f"balloon_{i:03d}.png"
                cv2.imwrite(str(balloon_path), balloon['image'])
            
            print(f"  Saved {len(all_balloons)} balloons to: {balloons_dir}")
        
        # 結果情報
        result = {
            'input_path': str(image_path),
            'output_path': str(output_path) if output_path else None,
            'balloons_dir': str(balloons_dir) if all_balloons else None,
            'num_panels': len(all_panels),
            'num_balloons': len(all_balloons),
            'is_spread_page': self.is_spread_page(image)
        }
        
        # メタデータを保存
        meta_path = output_dir / f"{image_path.stem}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Convert spread manga pages to vertical format')
    
    # 入出力設定
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    
    # モデル設定
    parser.add_argument('--panel-model', type=str, default=DEFAULT_PANEL_MODEL_PATH,
                        help='Path to panel segmentation model')
    parser.add_argument('--panel-model-type', type=str, default=DEFAULT_PANEL_MODEL_TYPE,
                        choices=['maskrcnn', 'mask2former'],
                        help='Panel model type')
    parser.add_argument('--balloon-model', type=str, default=None,
                        help='Path to balloon segmentation model (optional)')
    parser.add_argument('--input-type', type=str, default=DEFAULT_INPUT_TYPE,
                        choices=['gray', '3ch'],
                        help='Input type for panel model')
    
    # 処理パラメータ
    parser.add_argument('--score-threshold', type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help='Score threshold for panel detection')
    parser.add_argument('--panel-margin', type=int, default=DEFAULT_PANEL_MARGIN,
                        help='Margin between panels in vertical manga')
    parser.add_argument('--img-size', type=int, nargs=2, default=list(DEFAULT_IMG_SIZE),
                        help='Model input size (H W)')
    
    args = parser.parse_args()
    
    # 変換器の初期化
    converter = MangaPage2Vertical(
        panel_model_path=args.panel_model,
        panel_model_type=args.panel_model_type,
        balloon_model_path=args.balloon_model,
        input_type=args.input_type,
        img_size=tuple(args.img_size),
        score_threshold=args.score_threshold,
        panel_margin=args.panel_margin
    )
    
    # 入力画像のリストを取得
    input_path = Path(args.input)
    if input_path.is_dir():
        image_paths = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.jpeg')) + \
                      list(input_path.glob('*.png'))
    else:
        image_paths = [input_path]
    
    if not image_paths:
        print(f"No images found in {args.input}")
        return
    
    print(f"Found {len(image_paths)} image(s)")
    
    # 各画像を処理
    results = []
    for image_path in image_paths:
        try:
            result = converter.convert(str(image_path), args.output)
            results.append(result)
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 全体の結果を保存
    summary_path = Path(args.output) / "conversion_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Processed {len(results)} image(s)")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
