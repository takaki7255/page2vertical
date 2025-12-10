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
import tempfile

# predict.pyからインポート
from predict import (
    create_maskrcnn,
    create_mask2former,
    load_image,
    predict_maskrcnn,
    predict_mask2former,
)

# panel_order_estimater.pyからインポート
from panel_order_estimator import panel_order_estimater

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
DEFAULT_BALLOON_MODEL_PATH = "./balloon_models/real3000_dataset-unet-01.pt"
DEFAULT_BALLOON_IMG_SIZE = (384, 512)  # (H, W)


# ================== U-Net モデル定義 ==================
class DoubleConv(torch.nn.Module):
    """U-Net用のダブルコンボリューションブロック"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, out_c, 3, 1, 1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, 3, 1, 1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(torch.nn.Module):
    """吹き出しセグメンテーション用U-Net"""
    def __init__(self, n_classes=1, chs=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.downs, in_c = torch.nn.ModuleList(), 3
        for c in chs[:-1]:
            self.downs.append(DoubleConv(in_c, c))
            in_c = c
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        self.ups_tr = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(chs[i], chs[i-1], 2, 2)
            for i in range(len(chs)-1, 0, -1)
        ])
        self.up_convs = torch.nn.ModuleList([
            DoubleConv(chs[i], chs[i-1])
            for i in range(len(chs)-1, 0, -1)
        ])
        self.out_conv = torch.nn.Conv2d(chs[0], n_classes, 1)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for l in self.downs:
            x = l(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, sk in zip(self.ups_tr, self.up_convs, skips[::-1]):
            x = up(x)
            x = torch.cat([x, sk], 1)
            x = conv(x)
        return self.out_conv(x)


class BalloonSegmenter:
    """
    吹き出しのセマンティックセグメンテーション
    U-Netベースのモデルを使用
    """
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        device: torch.device = None,
        img_size: Tuple[int, int] = DEFAULT_BALLOON_IMG_SIZE
    ):
        self.device = device or torch.device('cpu')
        self.model = None
        self.model_path = model_path
        self.img_size = img_size  # (H, W)
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            print(f"Balloon segmentation model loaded: {model_path}")
        else:
            print("No balloon model specified. Using simple detection fallback.")
    
    def _load_model(self, model_path: str):
        """U-Netモデルを読み込む"""
        self.model = UNet(n_classes=1).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
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
        
        # U-Netによるセグメンテーション
        return self._unet_segmentation(image)
    
    def _unet_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        U-Netモデルによる吹き出しセグメンテーション
        
        Args:
            image: BGR画像 (H, W, 3)
        
        Returns:
            mask: バイナリマスク (H, W) uint8
        """
        orig_h, orig_w = image.shape[:2]
        
        # BGR → RGB → PIL → Tensor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # リサイズ（モデルの入力サイズに合わせる）
        image_resized = image_pil.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        
        # Tensor化 [0, 1]
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        image_tensor = image_tensor.to(self.device)
        
        # 推論
        with torch.no_grad():
            pred = torch.sigmoid(self.model(image_tensor))
            pred_binary = (pred > 0.5).float()
        
        # マスクを元画像サイズにリサイズ
        mask_np = pred_binary[0, 0].cpu().numpy()
        mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        
        return mask_uint8
    
    def _simple_balloon_detection(self, image: np.ndarray) -> np.ndarray:
        """
        簡易的な吹き出し検出（白色領域をベースに検出）
        U-Netモデルがない場合のフォールバック
        
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
        balloon_model_path: Optional[str] = DEFAULT_BALLOON_MODEL_PATH,
        balloon_img_size: Tuple[int, int] = DEFAULT_BALLOON_IMG_SIZE,
        input_type: str = DEFAULT_INPUT_TYPE,
        img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        panel_margin: int = DEFAULT_PANEL_MARGIN,
        smooth_mask: bool = False,
        smooth_kernel_size: int = 5,
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
        self.smooth_mask = smooth_mask
        self.smooth_kernel_size = smooth_kernel_size
        
        # コマ検出モデルの読み込み
        self._load_panel_model(panel_model_path, panel_model_type)
        
        # 吹き出しセグメンテーション
        self.balloon_segmenter = BalloonSegmenter(
            model_path=balloon_model_path, 
            device=self.device,
            img_size=balloon_img_size
        )
        
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
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # state_dictのキーに "model." プレフィックスがある場合は削除
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        
        self.panel_model.load_state_dict(state_dict)
        self.panel_model = self.panel_model.to(self.device)
        self.panel_model.eval()
        
        print("Model loaded successfully.")
    
    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        モルフォロジー処理でマスクの境界をスムーズにする
        
        Args:
            mask: バイナリマスク (H, W) uint8
        
        Returns:
            smoothed_mask: スムージング後のマスク
        """
        # マスクが空の場合はそのまま返す
        if mask.sum() == 0:
            return mask
        
        # カーネルサイズに基づいて楕円形カーネルを作成
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.smooth_kernel_size, self.smooth_kernel_size)
        )
        
        # Closing（穴埋め）→ Opening（小さなノイズ除去）の順で処理
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        return smoothed
    
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
            temp_path = os.path.join(tempfile.gettempdir(), "temp_panel_detect.jpg")
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
            
            # マスクのスムージング処理
            if self.smooth_mask:
                mask_resized = self._smooth_mask(mask_resized)
            
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
        gap_ratio_w: float = 0.56,  # 画像の横幅に対するコマ間余白の比
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        コマを縦に並べて縦読み漫画を作成
        
        Args:
            panels: コマ画像のリスト（読み順）
            margin: コマ間の余白（Noneの場合はgap_ratio_wから自動計算）
            gap_ratio_w: 画像の横幅に対するコマ間余白の比（デフォルト: 0.56）
            background_color: 背景色（BGR）
        
        Returns:
            vertical_manga: 縦読み漫画画像
        """
        if not panels:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # 出力画像の横幅（最大コマ幅）
        max_width = max(panel.shape[1] for panel in panels)

        # margin が指定されていなければ、横幅に対する比から自動計算
        if margin is None:
            margin = int(max_width * gap_ratio_w)

        num_panels = len(panels)

        # 上下の余白も同じ margin にしておく
        top_margin = margin
        bottom_margin = margin

        total_height = (
            sum(p.shape[0] for p in panels)
            + margin * (num_panels - 1)  # コマとコマの「間」の余白
            + top_margin + bottom_margin
        )

        vertical_manga = np.full(
            (total_height, max_width, 3),
            background_color,
            dtype=np.uint8
        )

        current_y = top_margin
        for panel in panels:
            h, w = panel.shape[:2]
            x_offset = (max_width - w) // 2
            vertical_manga[current_y:current_y+h, x_offset:x_offset+w] = panel
            current_y += h + margin  # ここが「コマ間余白」

        return vertical_manga
    
    def process_page(self, image: np.ndarray, image_path: Optional[str] = None) -> Tuple[List[np.ndarray], List[Dict], List[np.ndarray], List[Dict], np.ndarray]:
        """
        単一ページを処理
        
        Returns:
            (ordered_panels_images, balloons, masks, panels_info, balloon_mask)
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
            return [], balloons, [], [], balloon_mask
        
        # 5. コマの順序推定
        ordered_panels = self.estimate_panel_order(panels_info, w, h)
        
        # 6. 順序に対応するマスクを並び替え
        # panels_infoのidとordered_panelsのidを対応させる
        id_to_mask = {info['id']: mask for info, mask in zip(panels_info, masks)}
        ordered_masks = [id_to_mask[panel['id']] for panel in ordered_panels]
        
        # 7. 順序通りにコマを切り出し
        ordered_panel_images = []
        for panel in ordered_panels:
            panel_image = self.extract_panel_image(inpainted_image, panel)
            ordered_panel_images.append(panel_image)
        
        return ordered_panel_images, balloons, ordered_masks, ordered_panels, balloon_mask
    
    def create_comparison_image(
        self, 
        original: np.ndarray, 
        masks: List[np.ndarray],
        panels_info: List[Dict],
        balloon_mask: np.ndarray
    ) -> np.ndarray:
        """
        元画像とセグメンテーション結果の比較画像を作成
        
        Args:
            original: 元画像（BGR）
            masks: コママスクのリスト
            panels_info: コマ情報のリスト
            balloon_mask: 吹き出しマスク
        
        Returns:
            comparison: 比較画像
        """
        h, w = original.shape[:2]
        
        # オーバーレイ画像を作成
        overlay = original.copy()
        
        # コママスクを色付きでオーバーレイ
        np.random.seed(42)
        colors = np.random.randint(50, 255, (len(masks), 3)).tolist()
        
        for i, (mask, info) in enumerate(zip(masks, panels_info)):
            color = colors[i % len(colors)]
            # マスクのサイズが異なる場合はリサイズ
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_colored = np.zeros_like(original)
            mask_colored[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.4, 0)
            
            # バウンディングボックスと番号を描画
            xmin, ymin = int(info['xmin']), int(info['ymin'])
            xmax, ymax = int(info['xmax']), int(info['ymax'])
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
            
            # パネル番号を描画
            label = f"#{i+1}"
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(overlay, (xmin, ymin - text_h - 10), (xmin + text_w + 10, ymin), color, -1)
            cv2.putText(overlay, label, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # 吹き出しマスクのサイズが異なる場合はリサイズ
        if balloon_mask.shape[:2] != (h, w):
            balloon_mask = cv2.resize(balloon_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 吹き出しマスクを青でオーバーレイ
        balloon_colored = np.zeros_like(original)
        balloon_colored[balloon_mask > 0] = [255, 100, 100]  # 青系
        overlay = cv2.addWeighted(overlay, 1.0, balloon_colored, 0.3, 0)
        
        # 元画像とオーバーレイを横に並べる
        comparison = np.concatenate([original, overlay], axis=1)
        
        # ラベルを追加
        label_h = 40
        label_bar = np.full((label_h, comparison.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(label_bar, "Original", (w // 2 - 50, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(label_bar, "Segmentation Result", (w + w // 2 - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        comparison = np.concatenate([label_bar, comparison], axis=0)
        
        return comparison
    
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
        
        # 画像ごとの出力ディレクトリを作成
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        panels_dir = image_output_dir / "panels"
        balloons_dir = image_output_dir / "balloons"
        panels_dir.mkdir(exist_ok=True)
        balloons_dir.mkdir(exist_ok=True)
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Processing: {image_path.name}")
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
        
        all_panels = []
        all_balloons = []
        all_masks = []
        all_panels_info = []
        all_balloon_masks = []
        page_images = []  # 比較画像用
        
        # 見開きページかチェック
        if self.is_spread_page(image):
            print("  Detected as spread page. Splitting...")
            
            # 右ページ（先）と左ページ（後）に分割
            right_page, left_page = self.split_spread_page(image)
            
            # 右ページを処理
            print("  Processing right page...")
            right_temp_path = os.path.join(tempfile.gettempdir(), "temp_right_page.jpg")
            cv2.imwrite(right_temp_path, right_page)
            right_panels, right_balloons, right_masks, right_info, right_balloon_mask = self.process_page(right_page, right_temp_path)
            all_panels.extend(right_panels)
            all_balloons.extend(right_balloons)
            all_masks.extend(right_masks)
            all_panels_info.extend(right_info)
            all_balloon_masks.append(right_balloon_mask)
            page_images.append(("right_page", right_page, right_masks, right_info, right_balloon_mask))
            
            # 左ページを処理
            print("  Processing left page...")
            left_temp_path = os.path.join(tempfile.gettempdir(), "temp_left_page.jpg")
            cv2.imwrite(left_temp_path, left_page)
            left_panels, left_balloons, left_masks, left_info, left_balloon_mask = self.process_page(left_page, left_temp_path)
            all_panels.extend(left_panels)
            all_balloons.extend(left_balloons)
            all_masks.extend(left_masks)
            all_panels_info.extend(left_info)
            all_balloon_masks.append(left_balloon_mask)
            page_images.append(("left_page", left_page, left_masks, left_info, left_balloon_mask))
        else:
            print("  Detected as single page.")
            panels, balloons, masks, panels_info, balloon_mask = self.process_page(image, str(image_path))
            all_panels.extend(panels)
            all_balloons.extend(balloons)
            all_masks.extend(masks)
            all_panels_info.extend(panels_info)
            all_balloon_masks.append(balloon_mask)
            page_images.append(("page", image, masks, panels_info, balloon_mask))
        
        print(f"  Total panels: {len(all_panels)}")
        print(f"  Total balloons: {len(all_balloons)}")
        
        # 元画像をコピー
        original_path = image_output_dir / f"original{image_path.suffix}"
        cv2.imwrite(str(original_path), image)
        
        # コマを個別に保存
        for i, panel in enumerate(all_panels):
            panel_path = panels_dir / f"panel_{i:03d}.png"
            cv2.imwrite(str(panel_path), panel)
        print(f"  Saved {len(all_panels)} panels to: {panels_dir}")
        
        # 吹き出しを個別に保存
        for i, balloon in enumerate(all_balloons):
            balloon_path = balloons_dir / f"balloon_{i:03d}.png"
            cv2.imwrite(str(balloon_path), balloon['image'])
        print(f"  Saved {len(all_balloons)} balloons to: {balloons_dir}")
        
        # 縦読み漫画を作成
        if all_panels:
            vertical_manga = self.create_vertical_manga(all_panels)
            output_path = image_output_dir / "vertical.png"
            cv2.imwrite(str(output_path), vertical_manga)
            print(f"  Saved vertical manga: {output_path}")
        else:
            output_path = None
            print("  No panels to create vertical manga.")
        
        # 比較画像を作成
        for page_name, page_img, masks, info, balloon_mask in page_images:
            if masks and info:
                comparison = self.create_comparison_image(page_img, masks, info, balloon_mask)
                comparison_path = image_output_dir / f"comparison_{page_name}.png"
                cv2.imwrite(str(comparison_path), comparison)
                print(f"  Saved comparison image: {comparison_path}")
        
        # 結果情報
        result = {
            'input_path': str(image_path),
            'output_dir': str(image_output_dir),
            'vertical_path': str(output_path) if output_path else None,
            'panels_dir': str(panels_dir),
            'balloons_dir': str(balloons_dir),
            'num_panels': len(all_panels),
            'num_balloons': len(all_balloons),
            'is_spread_page': self.is_spread_page(image)
        }
        
        # メタデータを保存
        meta_path = image_output_dir / "meta.json"
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
    parser.add_argument('--balloon-model', type=str, default=DEFAULT_BALLOON_MODEL_PATH,
                        help='Path to balloon segmentation U-Net model')
    parser.add_argument('--balloon-img-size', type=int, nargs=2, default=list(DEFAULT_BALLOON_IMG_SIZE),
                        help='Balloon model input size (H W)')
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
    
    # マスクスムージング設定
    parser.add_argument('--smooth-mask', action='store_true',
                        help='Enable morphological smoothing for panel masks')
    parser.add_argument('--smooth-kernel-size', type=int, default=5,
                        help='Kernel size for mask smoothing (default: 5)')
    
    args = parser.parse_args()
    
    # 変換器の初期化
    converter = MangaPage2Vertical(
        panel_model_path=args.panel_model,
        panel_model_type=args.panel_model_type,
        balloon_model_path=args.balloon_model,
        balloon_img_size=tuple(args.balloon_img_size),
        input_type=args.input_type,
        img_size=tuple(args.img_size),
        score_threshold=args.score_threshold,
        panel_margin=args.panel_margin,
        smooth_mask=args.smooth_mask,
        smooth_kernel_size=args.smooth_kernel_size
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
