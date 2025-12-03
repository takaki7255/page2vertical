from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation
import os
import random
import glob

simple_lama = SimpleLama()

# ベースディレクトリ
base_dir = "./../Manga109_released_2023_12_07/"
images_dir = os.path.join(base_dir, "images")
masks_dir = os.path.join(base_dir, "masks")

# 出力ディレクトリを作成
output_dir = "./experiment_results"
output_images_dir = os.path.join(output_dir, "images")
output_masks_dir = os.path.join(output_dir, "masks")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# 利用可能なコミックタイトルを取得
comic_titles = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]

# ランダムに画像を選択する関数
def get_random_images(num_images=100):
    selected_images = []
    
    for comic_title in comic_titles:
        # 画像ディレクトリとマスクディレクトリのパス
        comic_img_dir = os.path.join(images_dir, comic_title)
        comic_mask_dir = os.path.join(masks_dir, comic_title, "balloon")
        
        # 両方のディレクトリが存在するかチェック
        if not os.path.exists(comic_mask_dir):
            continue
            
        # 画像ファイルを取得
        img_files = glob.glob(os.path.join(comic_img_dir, "*.jpg"))
        
        for img_file in img_files:
            img_name = os.path.basename(img_file)
            img_base = os.path.splitext(img_name)[0]
            mask_file = os.path.join(comic_mask_dir, f"{img_base}_mask.png")
            
            # 対応するマスクファイルが存在するかチェック
            if os.path.exists(mask_file):
                selected_images.append({
                    'comic_title': comic_title,
                    'img_path': img_file,
                    'mask_path': mask_file,
                    'img_name': img_base
                })
    
    # ランダムに指定数選択
    return random.sample(selected_images, min(num_images, len(selected_images)))

# ランダムに100画像を選択
selected_images = get_random_images(100)

print(f"選択された画像数: {len(selected_images)}")

# 各画像に対してインペイント処理を実行
for i, img_info in enumerate(selected_images):
    try:
        print(f"処理中 ({i+1}/{len(selected_images)}): {img_info['comic_title']}/{img_info['img_name']}")
        
        # 画像を読み込み、RGBに変換（3チャンネル）
        image = Image.open(img_info['img_path']).convert('RGB')
        
        # マスクを読み込み、グレースケールに変換（1チャンネル）
        mask = Image.open(img_info['mask_path']).convert('L')
        
        # マスクをバイナリ化（0または255の値にする）
        mask_array = np.array(mask)
        # 閾値127で二値化し、255のピクセルがinpaintingされる領域
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # マスク領域に膨張処理を適用
        binary_mask = mask_array > 0  # バイナリマスクに変換
        dilated_mask = binary_dilation(binary_mask, iterations=3)  # 3回膨張処理
        mask_array = (dilated_mask * 255).astype(np.uint8)  # 0-255の値に戻す
        
        processed_mask = Image.fromarray(mask_array)
        
        # LAMAによるインペイント
        result = simple_lama(image, processed_mask)
        
        # ファイル名を生成
        filename_base = f"{img_info['comic_title']}_{img_info['img_name']}"
        
        # 元画像を保存
        original_image_path = os.path.join(output_images_dir, f"{filename_base}.png")
        image.save(original_image_path)
        
        # 処理済みマスクを保存
        mask_path = os.path.join(output_masks_dir, f"{filename_base}_mask.png")
        processed_mask.save(mask_path)
        
        # インペイント結果を保存
        inpainted_path = os.path.join(output_dir, f"{filename_base}_inpainted.png")
        result.save(inpainted_path)
        
        print(f"保存完了: {filename_base}")
        
    except Exception as e:
        print(f"エラーが発生しました ({img_info['comic_title']}/{img_info['img_name']}): {e}")

print("実験完了!")