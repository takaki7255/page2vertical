"""
Manga109のアノテーションを使ってpanel_order_estimator.pyをテストするスクリプト

Usage:
    python test_order_estimator.py
    python test_order_estimator.py --manga-dir /path/to/Manga109
    python test_order_estimator.py --visualize  # 可視化結果を保存
    python test_order_estimator.py --book ARMS --page 3  # 特定のページをテスト
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from panel_order_estimator import panel_order_estimater


# デフォルトのManga109ディレクトリ
DEFAULT_MANGA_DIR = "/Users/x20047xx/研究室/manga/Manga109_released_2023_12_07"


def parse_manga109_xml(xml_path: str) -> Dict:
    """
    Manga109のXMLアノテーションファイルをパース
    
    Args:
        xml_path: XMLファイルパス
    
    Returns:
        book_data: {'title': str, 'pages': [{'index': int, 'width': int, 'height': int, 'frames': [...]}]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    book_title = root.get('title')
    
    pages = []
    for page_elem in root.find('pages').findall('page'):
        page_data = {
            'index': int(page_elem.get('index')),
            'width': int(page_elem.get('width')),
            'height': int(page_elem.get('height')),
            'frames': []
        }
        
        # フレーム（コマ）情報を抽出
        for frame_elem in page_elem.findall('frame'):
            frame_data = {
                'type': 'frame',
                'id': frame_elem.get('id'),
                'xmin': frame_elem.get('xmin'),
                'ymin': frame_elem.get('ymin'),
                'xmax': frame_elem.get('xmax'),
                'ymax': frame_elem.get('ymax')
            }
            page_data['frames'].append(frame_data)
        
        pages.append(page_data)
    
    return {
        'title': book_title,
        'pages': pages
    }


def get_all_books(manga_dir: str) -> List[str]:
    """利用可能な全ての漫画タイトルを取得"""
    annotations_dir = os.path.join(manga_dir, 'annotations')
    books = []
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            books.append(xml_file.replace('.xml', ''))
    return sorted(books)


def get_image_path(manga_dir: str, book_title: str, page_index: int) -> str:
    """ページ画像のパスを取得"""
    images_dir = os.path.join(manga_dir, 'images', book_title)
    # Manga109の画像ファイル名は000.jpg, 001.jpg, ... の形式
    image_path = os.path.join(images_dir, f'{page_index:03d}.jpg')
    return image_path


def visualize_panel_order(
    image: np.ndarray,
    ordered_panels: List[Dict],
    output_path: str
) -> None:
    """
    コマ順序を可視化して保存
    
    Args:
        image: 元画像（BGR）
        ordered_panels: 順序付けされたコマリスト
        output_path: 出力パス
    """
    vis_image = image.copy()
    
    # カラーマップ（順序に応じて色を変える）
    num_panels = len(ordered_panels)
    colors = []
    for i in range(num_panels):
        # 虹色グラデーション
        hue = int(180 * i / max(num_panels, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    
    for i, panel in enumerate(ordered_panels):
        xmin = int(panel['xmin'])
        ymin = int(panel['ymin'])
        xmax = int(panel['xmax'])
        ymax = int(panel['ymax'])
        
        color = colors[i]
        
        # バウンディングボックスを描画
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 3)
        
        # 順序番号を描画
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        
        # 番号の背景（見やすくするため）
        text = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 4
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(
            vis_image,
            (center_x - text_w // 2 - 10, center_y - text_h // 2 - 10),
            (center_x + text_w // 2 + 10, center_y + text_h // 2 + 10),
            (255, 255, 255),
            -1
        )
        cv2.putText(
            vis_image,
            text,
            (center_x - text_w // 2, center_y + text_h // 2),
            font,
            font_scale,
            color,
            thickness
        )
        
        # 矢印で順序を表示（次のコマへ）
        if i < num_panels - 1:
            next_panel = ordered_panels[i + 1]
            next_center_x = (int(next_panel['xmin']) + int(next_panel['xmax'])) // 2
            next_center_y = (int(next_panel['ymin']) + int(next_panel['ymax'])) // 2
            
            cv2.arrowedLine(
                vis_image,
                (center_x, center_y),
                (next_center_x, next_center_y),
                (0, 0, 255),
                2,
                tipLength=0.05
            )
    
    cv2.imwrite(output_path, vis_image)
    print(f"  Saved visualization: {output_path}")


def display_panels_in_order(
    image: np.ndarray,
    ordered_panels: List[Dict],
    window_name: str = "Panel Order Viewer"
) -> None:
    """
    元の画像を表示しながら、推定した順序通りにコマを表示する
    
    Args:
        image: 元画像（BGR）
        ordered_panels: 順序付けされたコマリスト
        window_name: ウィンドウ名
    
    操作方法:
        - スペースキー / n: 次のコマ
        - p: 前のコマ
        - r: 最初に戻る
        - q / ESC: 終了
    """
    if not ordered_panels:
        print("No panels to display.")
        return
    
    num_panels = len(ordered_panels)
    current_idx = 0
    
    # 元画像のウィンドウサイズを調整（大きすぎる場合）
    h, w = image.shape[:2]
    max_size = 800
    scale = min(max_size / w, max_size / h, 1.0)
    display_w = int(w * scale)
    display_h = int(h * scale)
    
    # コマ表示用のウィンドウサイズ
    panel_window_size = 400
    
    print("\n=== Panel Order Viewer ===")
    print("Controls:")
    print("  Space / n: Next panel")
    print("  p: Previous panel")
    print("  r: Reset to first panel")
    print("  q / ESC: Quit")
    print("==========================\n")
    
    # ウィンドウを作成
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow("Current Panel", cv2.WINDOW_NORMAL)
    
    while True:
        # 元画像に現在のコマをハイライト
        vis_image = image.copy()
        
        # すべてのコマを表示
        for i, panel in enumerate(ordered_panels):
            xmin = int(panel['xmin'])
            ymin = int(panel['ymin'])
            xmax = int(panel['xmax'])
            ymax = int(panel['ymax'])
            
            if i < current_idx:
                # 既に見たコマは緑で表示
                cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 200, 0), 2)
            elif i == current_idx:
                # 現在のコマは赤で強調
                cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
            else:
                # まだ見ていないコマはグレー
                cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), (128, 128, 128), 1)
            
            # 順序番号を表示
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            text = str(i + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            if i == current_idx:
                font_scale = 1.5
                thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(
                    vis_image,
                    (center_x - text_w // 2 - 5, center_y - text_h // 2 - 5),
                    (center_x + text_w // 2 + 5, center_y + text_h // 2 + 5),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    vis_image, text,
                    (center_x - text_w // 2, center_y + text_h // 2),
                    font, font_scale, (255, 255, 255), thickness
                )
            else:
                font_scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(
                    vis_image,
                    (center_x - text_w // 2 - 3, center_y - text_h // 2 - 3),
                    (center_x + text_w // 2 + 3, center_y + text_h // 2 + 3),
                    (255, 255, 255),
                    -1
                )
                color = (0, 200, 0) if i < current_idx else (128, 128, 128)
                cv2.putText(
                    vis_image, text,
                    (center_x - text_w // 2, center_y + text_h // 2),
                    font, font_scale, color, thickness
                )
        
        # 情報テキストを追加
        info_text = f"Panel {current_idx + 1} / {num_panels}"
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 元画像をリサイズして表示
        vis_resized = cv2.resize(vis_image, (display_w, display_h))
        cv2.imshow(window_name, vis_resized)
        
        # 現在のコマを切り出して表示
        panel = ordered_panels[current_idx]
        xmin = int(panel['xmin'])
        ymin = int(panel['ymin'])
        xmax = int(panel['xmax'])
        ymax = int(panel['ymax'])
        
        panel_image = image[ymin:ymax, xmin:xmax].copy()
        
        # コマ画像をリサイズ（アスペクト比を維持）
        ph, pw = panel_image.shape[:2]
        if pw > 0 and ph > 0:
            panel_scale = min(panel_window_size / pw, panel_window_size / ph, 1.0)
            panel_display_w = max(int(pw * panel_scale), 1)
            panel_display_h = max(int(ph * panel_scale), 1)
            
            panel_resized = cv2.resize(panel_image, (panel_display_w, panel_display_h))
            
            # コマ番号を表示
            panel_info = f"Panel {current_idx + 1}"
            cv2.putText(panel_resized, panel_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(panel_resized, panel_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Current Panel", panel_resized)
        
        # キー入力待ち（100msごとにチェック）
        key = cv2.waitKey(0)
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' ') or key == ord('n'):  # Space or n
            current_idx = min(current_idx + 1, num_panels - 1)
        elif key == ord('p'):  # p
            current_idx = max(current_idx - 1, 0)
        elif key == ord('r'):  # Reset
            current_idx = 0
    
    cv2.destroyAllWindows()


def test_single_page(
    manga_dir: str,
    book_title: str,
    page_index: int,
    visualize: bool = False,
    output_dir: Optional[str] = None,
    display: bool = False
) -> Dict:
    """
    単一ページのコマ順序推定をテスト
    
    Returns:
        result: {'book': str, 'page': int, 'num_panels': int, 'success': bool, 'ordered_ids': [...]}
    """
    xml_path = os.path.join(manga_dir, 'annotations', f'{book_title}.xml')
    book_data = parse_manga109_xml(xml_path)
    
    # 該当ページを探す
    page_data = None
    for page in book_data['pages']:
        if page['index'] == page_index:
            page_data = page
            break
    
    if page_data is None:
        return {
            'book': book_title,
            'page': page_index,
            'num_panels': 0,
            'success': False,
            'error': 'Page not found'
        }
    
    frames = page_data['frames']
    
    if len(frames) == 0:
        return {
            'book': book_title,
            'page': page_index,
            'num_panels': 0,
            'success': True,
            'ordered_ids': [],
            'message': 'No frames in this page'
        }
    
    # コマ順序推定を実行
    try:
        # コピーを渡す（元データを変更しないため）
        frames_copy = [f.copy() for f in frames]
        ordered_panels = panel_order_estimater(
            frames_copy,
            page_data['width'],
            page_data['height']
        )
        
        ordered_ids = [p['id'] for p in ordered_panels]
        
        result = {
            'book': book_title,
            'page': page_index,
            'num_panels': len(frames),
            'success': True,
            'ordered_ids': ordered_ids,
            'width': page_data['width'],
            'height': page_data['height']
        }
        
        # 画像を読み込む（可視化または表示用）
        image_path = get_image_path(manga_dir, book_title, page_index)
        image = None
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
        
        # 可視化（ファイル保存）
        if visualize and output_dir and image is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f'{book_title}_page{page_index:03d}_order.jpg'
            )
            visualize_panel_order(image, ordered_panels, output_path)
        
        # 画面表示
        if display and image is not None:
            window_name = f"{book_title} - Page {page_index}"
            # 各パネルのバウンディングボックスを出力
            print("\nPanel bounding boxes (in order):")
            for i, p in enumerate(ordered_panels):
                print(f"  {i+1}. id={p['id']}: ({p['xmin']}, {p['ymin']}) - ({p['xmax']}, {p['ymax']})")
            display_panels_in_order(image, ordered_panels, window_name)
        
        return result
        
    except Exception as e:
        return {
            'book': book_title,
            'page': page_index,
            'num_panels': len(frames),
            'success': False,
            'error': str(e)
        }


def test_random_pages(
    manga_dir: str,
    num_samples: int = 50,
    visualize: bool = False,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> List[Dict]:
    """
    ランダムにページを選んでテスト
    
    Args:
        manga_dir: Manga109ディレクトリ
        num_samples: サンプル数
        visualize: 可視化するか
        output_dir: 可視化画像の出力ディレクトリ
        seed: 乱数シード
    
    Returns:
        results: テスト結果のリスト
    """
    random.seed(seed)
    
    books = get_all_books(manga_dir)
    print(f"Found {len(books)} books")
    
    # 全ページからサンプリング候補を収集
    candidates = []
    for book in books:
        xml_path = os.path.join(manga_dir, 'annotations', f'{book}.xml')
        book_data = parse_manga109_xml(xml_path)
        
        for page in book_data['pages']:
            # フレームがあるページのみ
            if len(page['frames']) >= 2:  # 2コマ以上
                candidates.append({
                    'book': book,
                    'page': page['index'],
                    'num_frames': len(page['frames'])
                })
    
    print(f"Found {len(candidates)} pages with 2+ panels")
    
    # サンプリング
    samples = random.sample(candidates, min(num_samples, len(candidates)))
    
    results = []
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Testing {sample['book']} page {sample['page']} ({sample['num_frames']} panels)")
        
        result = test_single_page(
            manga_dir,
            sample['book'],
            sample['page'],
            visualize=visualize,
            output_dir=output_dir
        )
        results.append(result)
        
        if result['success']:
            print(f"  Order: {result['ordered_ids']}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    return results


def print_summary(results: List[Dict]) -> None:
    """テスト結果のサマリーを表示"""
    total = len(results)
    success = sum(1 for r in results if r['success'])
    failed = total - success
    
    total_panels = sum(r['num_panels'] for r in results)
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Total pages tested: {total}")
    print(f"Successful: {success} ({100*success/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"Total panels processed: {total_panels}")
    
    if failed > 0:
        print("\nFailed cases:")
        for r in results:
            if not r['success']:
                print(f"  - {r['book']} page {r['page']}: {r.get('error', 'Unknown')}")


def test_specific_case():
    """panel_order_estimater.pyにあるテストケースを実行"""
    print("Testing with the example from panel_order_estimater.py...")
    print("This is ARMS page 3")
    
    data = [
        {'type': 'frame', 'id': '00000009', 'xmin': '899', 'ymin': '585', 'xmax': '1170', 'ymax': '1085'},
        {'type': 'frame', 'id': '0000000c', 'xmin': '2', 'ymin': '0', 'xmax': '826', 'ymax': '513'},
        {'type': 'frame', 'id': '0000000e', 'xmin': '72', 'ymin': '516', 'xmax': '743', 'ymax': '1101'},
        {'type': 'frame', 'id': '00000014', 'xmin': '906', 'ymin': '95', 'xmax': '1575', 'ymax': '576'},
        {'type': 'frame', 'id': '0000001d', 'xmin': '1167', 'ymin': '588', 'xmax': '1580', 'ymax': '1090'}
    ]
    
    # ARMS page 3のサイズ
    img_width = 1654
    img_height = 1170
    
    print(f"\nInput panels (unordered):")
    for p in data:
        print(f"  {p['id']}: ({p['xmin']}, {p['ymin']}) - ({p['xmax']}, {p['ymax']})")
    
    result = panel_order_estimater(data.copy(), img_width, img_height)
    
    print(f"\nEstimated order:")
    for i, p in enumerate(result):
        print(f"  {i+1}. {p['id']}: ({p['xmin']}, {p['ymin']}) - ({p['xmax']}, {p['ymax']})")
    
    # 日本語漫画の読み順（右上から左下）に基づく期待順序
    # 00000014: 右上 (906-1575, 95-576)
    # 0000000c: 左上 (2-826, 0-513)
    # 0000001d: 右下上 (1167-1580, 588-1090)
    # 00000009: 右下左 (899-1170, 585-1085)
    # 0000000e: 左下 (72-743, 516-1101)
    expected_order = ['00000014', '0000000c', '0000001d', '00000009', '0000000e']
    
    print(f"\nExpected order (manual): {expected_order}")
    actual_order = [p['id'] for p in result]
    print(f"Actual order: {actual_order}")
    
    if expected_order == actual_order:
        print("\n✓ Test PASSED!")
    else:
        print("\n✗ Test FAILED - order mismatch")


def main():
    parser = argparse.ArgumentParser(description='Test panel order estimator with Manga109')
    
    parser.add_argument('--manga-dir', type=str, default=DEFAULT_MANGA_DIR,
                        help='Path to Manga109 directory')
    parser.add_argument('--book', type=str, default=None,
                        help='Specific book title to test')
    parser.add_argument('--page', type=int, default=None,
                        help='Specific page index to test')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of random samples to test')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--display', action='store_true',
                        help='Display panels in order interactively')
    parser.add_argument('--output-dir', type=str, default='./test_order_results',
                        help='Output directory for visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--test-example', action='store_true',
                        help='Test with the example from panel_order_estimater.py')
    
    args = parser.parse_args()
    
    # 組み込みテストケースを実行
    if args.test_example:
        test_specific_case()
        print()
    
    # Manga109が存在するか確認
    if not os.path.exists(args.manga_dir):
        print(f"Manga109 directory not found: {args.manga_dir}")
        print("Please specify the correct path with --manga-dir")
        return
    
    # 特定のページをテスト
    if args.book and args.page is not None:
        print(f"\nTesting {args.book} page {args.page}...")
        result = test_single_page(
            args.manga_dir,
            args.book,
            args.page,
            visualize=args.visualize,
            output_dir=args.output_dir,
            display=args.display
        )
        
        if result['success']:
            print(f"Order: {result['ordered_ids']}")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")
        return
    
    # ランダムテスト
    print(f"\nRunning random tests with {args.num_samples} samples...")
    results = test_random_pages(
        args.manga_dir,
        num_samples=args.num_samples,
        visualize=args.visualize,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print_summary(results)


if __name__ == '__main__':
    main()
