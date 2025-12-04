"""
Manga109からテスト用の画像を収集してフォルダを作成するスクリプト

Usage:
    # ランダムに50枚収集
    python collect_test_images.py --num 50 --output ./test_images
    
    # 特定の漫画から収集
    python collect_test_images.py --num 20 --book ARMS --output ./test_images
    
    # コマが多いページのみ収集（3コマ以上）
    python collect_test_images.py --num 50 --min-panels 3 --output ./test_images
    
    # 全ての漫画から各1枚ずつ収集
    python collect_test_images.py --one-per-book --output ./test_images
"""

import argparse
import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional


# デフォルトのManga109ディレクトリ
DEFAULT_MANGA_DIR = "/Users/x20047xx/研究室/manga/Manga109_released_2023_12_07"


def parse_manga109_xml(xml_path: str) -> Dict:
    """
    Manga109のXMLアノテーションファイルをパース
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
            'num_frames': len(page_elem.findall('frame'))
        }
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
    image_path = os.path.join(images_dir, f'{page_index:03d}.jpg')
    return image_path


def collect_candidates(
    manga_dir: str,
    book: Optional[str] = None,
    min_panels: int = 1,
    max_panels: Optional[int] = None
) -> List[Dict]:
    """
    収集候補となるページ情報を取得
    
    Args:
        manga_dir: Manga109ディレクトリ
        book: 特定の漫画タイトル（Noneなら全て）
        min_panels: 最小コマ数
        max_panels: 最大コマ数（Noneなら制限なし）
    
    Returns:
        candidates: [{'book': str, 'page': int, 'num_frames': int, 'width': int, 'height': int}, ...]
    """
    if book:
        books = [book]
    else:
        books = get_all_books(manga_dir)
    
    candidates = []
    for book_title in books:
        xml_path = os.path.join(manga_dir, 'annotations', f'{book_title}.xml')
        if not os.path.exists(xml_path):
            print(f"Warning: {xml_path} not found, skipping.")
            continue
        
        book_data = parse_manga109_xml(xml_path)
        
        for page in book_data['pages']:
            num_frames = page['num_frames']
            
            # コマ数でフィルタリング
            if num_frames < min_panels:
                continue
            if max_panels is not None and num_frames > max_panels:
                continue
            
            # 画像ファイルが存在するか確認
            image_path = get_image_path(manga_dir, book_title, page['index'])
            if not os.path.exists(image_path):
                continue
            
            candidates.append({
                'book': book_title,
                'page': page['index'],
                'num_frames': num_frames,
                'width': page['width'],
                'height': page['height'],
                'image_path': image_path
            })
    
    return candidates


def collect_images(
    manga_dir: str,
    output_dir: str,
    num_images: int = 50,
    book: Optional[str] = None,
    min_panels: int = 1,
    max_panels: Optional[int] = None,
    one_per_book: bool = False,
    seed: int = 42,
    copy_annotations: bool = False
) -> List[Dict]:
    """
    画像を収集してフォルダにコピー
    
    Args:
        manga_dir: Manga109ディレクトリ
        output_dir: 出力ディレクトリ
        num_images: 収集する画像数
        book: 特定の漫画タイトル
        min_panels: 最小コマ数
        max_panels: 最大コマ数
        one_per_book: 各漫画から1枚ずつ収集
        seed: 乱数シード
        copy_annotations: アノテーションファイルもコピーするか
    
    Returns:
        collected: 収集した画像の情報リスト
    """
    random.seed(seed)
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 候補を収集
    candidates = collect_candidates(manga_dir, book, min_panels, max_panels)
    
    if not candidates:
        print("No candidates found with the specified criteria.")
        return []
    
    print(f"Found {len(candidates)} candidate pages")
    
    # 選択
    if one_per_book:
        # 各漫画から1枚ずつ選択
        books_dict = {}
        for c in candidates:
            if c['book'] not in books_dict:
                books_dict[c['book']] = []
            books_dict[c['book']].append(c)
        
        selected = []
        for book_title, pages in books_dict.items():
            selected.append(random.choice(pages))
        
        # num_imagesより多い場合はさらにサンプリング
        if len(selected) > num_images:
            selected = random.sample(selected, num_images)
    else:
        # ランダムにサンプリング
        selected = random.sample(candidates, min(num_images, len(candidates)))
    
    print(f"Selected {len(selected)} images")
    
    # コピー
    collected = []
    for i, item in enumerate(selected):
        src_path = item['image_path']
        
        # ファイル名を生成（重複防止のため）
        filename = f"{item['book']}_{item['page']:03d}.jpg"
        dst_path = output_path / filename
        
        # コピー
        shutil.copy2(src_path, dst_path)
        
        item['output_path'] = str(dst_path)
        collected.append(item)
        
        print(f"  [{i+1}/{len(selected)}] Copied: {filename} ({item['num_frames']} panels)")
    
    # アノテーションファイルもコピー
    if copy_annotations:
        annotations_output = output_path / 'annotations'
        annotations_output.mkdir(exist_ok=True)
        
        copied_books = set()
        for item in collected:
            if item['book'] not in copied_books:
                src_xml = os.path.join(manga_dir, 'annotations', f"{item['book']}.xml")
                dst_xml = annotations_output / f"{item['book']}.xml"
                if os.path.exists(src_xml):
                    shutil.copy2(src_xml, dst_xml)
                    copied_books.add(item['book'])
        
        print(f"\nCopied {len(copied_books)} annotation files to {annotations_output}")
    
    # メタ情報を保存
    meta_path = output_path / 'collection_info.txt'
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"Collection Info\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total images: {len(collected)}\n")
        f.write(f"Min panels: {min_panels}\n")
        f.write(f"Max panels: {max_panels if max_panels else 'No limit'}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\n")
        f.write(f"Images:\n")
        for item in collected:
            f.write(f"  {item['book']}_{item['page']:03d}.jpg - {item['num_frames']} panels, {item['width']}x{item['height']}\n")
    
    print(f"\nCollection info saved to: {meta_path}")
    
    return collected


def main():
    parser = argparse.ArgumentParser(description='Collect test images from Manga109')
    
    parser.add_argument('--manga-dir', type=str, default=DEFAULT_MANGA_DIR,
                        help='Path to Manga109 directory')
    parser.add_argument('--output', '-o', type=str, default='./test_images',
                        help='Output directory')
    parser.add_argument('--num', '-n', type=int, default=50,
                        help='Number of images to collect')
    parser.add_argument('--book', '-b', type=str, default=None,
                        help='Specific book title to collect from')
    parser.add_argument('--min-panels', type=int, default=1,
                        help='Minimum number of panels in a page')
    parser.add_argument('--max-panels', type=int, default=None,
                        help='Maximum number of panels in a page')
    parser.add_argument('--one-per-book', action='store_true',
                        help='Collect one image per book')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--copy-annotations', action='store_true',
                        help='Also copy annotation XML files')
    parser.add_argument('--list-books', action='store_true',
                        help='List all available book titles and exit')
    
    args = parser.parse_args()
    
    # Manga109が存在するか確認
    if not os.path.exists(args.manga_dir):
        print(f"Manga109 directory not found: {args.manga_dir}")
        return
    
    # 漫画リストを表示して終了
    if args.list_books:
        books = get_all_books(args.manga_dir)
        print(f"Available books ({len(books)}):")
        for book in books:
            print(f"  {book}")
        return
    
    # 画像を収集
    collected = collect_images(
        manga_dir=args.manga_dir,
        output_dir=args.output,
        num_images=args.num,
        book=args.book,
        min_panels=args.min_panels,
        max_panels=args.max_panels,
        one_per_book=args.one_per_book,
        seed=args.seed,
        copy_annotations=args.copy_annotations
    )
    
    print(f"\nDone! Collected {len(collected)} images to {args.output}")


if __name__ == '__main__':
    main()
