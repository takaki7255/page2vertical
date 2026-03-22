"""
Run panel instance segmentation and balloon extraction/inpainting.

Outputs are saved separately for each input image:
- inpainted/no_balloons.png
- panel_segmentation/panels_overlay.png
- balloons/balloon_XXX.png

Usage examples:
    python segment_panels_and_balloons.py --input ./test_images --output ./pipeline_output
    python segment_panels_and_balloons.py --input ./test_images/page001.jpg --output ./pipeline_output
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from main import (
    DEFAULT_BALLOON_IMG_SIZE,
    DEFAULT_BALLOON_MODEL_PATH,
    DEFAULT_IMG_SIZE,
    DEFAULT_INPUT_TYPE,
    DEFAULT_PANEL_MODEL_PATH,
    DEFAULT_PANEL_MODEL_TYPE,
    DEFAULT_SCORE_THRESHOLD,
    MangaPage2Vertical,
)


def create_panel_overlay(image: np.ndarray, masks: List[np.ndarray], panels_info: List[Dict]) -> np.ndarray:
    """Create a simple visualization image for panel instance segmentation."""
    overlay = image.copy()

    # Stable colors for reproducible visualizations.
    np.random.seed(42)
    colors = np.random.randint(60, 255, (max(1, len(masks)), 3)).tolist()

    for idx, (mask, info) in enumerate(zip(masks, panels_info)):
        color = colors[idx % len(colors)]

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        tint = np.zeros_like(image)
        tint[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.45, 0)

        x1 = int(info["xmin"])
        y1 = int(info["ymin"])
        x2 = int(info["xmax"])
        y2 = int(info["ymax"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"#{idx + 1}",
            (x1 + 4, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def process_single_image(converter: MangaPage2Vertical, image_path: Path, output_root: Path) -> Dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

    image_out_dir = output_root / image_path.stem
    inpainted_dir = image_out_dir / "inpainted"
    panel_dir = image_out_dir / "panel_segmentation"
    balloons_dir = image_out_dir / "balloons"

    inpainted_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    balloons_dir.mkdir(parents=True, exist_ok=True)

    balloon_mask = converter.balloon_segmenter.segment(image)
    balloons = converter.extract_balloons(image, balloon_mask)
    inpainted = converter.inpainter.inpaint(image, balloon_mask)

    masks, panels_info = converter.detect_panels(inpainted, str(image_path))
    panel_overlay = create_panel_overlay(inpainted, masks, panels_info)

    # 1) Balloon-removed / inpainted image.
    inpainted_path = inpainted_dir / "no_balloons.png"
    cv2.imwrite(str(inpainted_path), inpainted)

    # 2) Panel instance segmentation visualization.
    panel_overlay_path = panel_dir / "panels_overlay.png"
    cv2.imwrite(str(panel_overlay_path), panel_overlay)

    # Optional panel masks for debugging or downstream processing.
    for idx, mask in enumerate(masks):
        mask_path = panel_dir / f"panel_mask_{idx:03d}.png"
        cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)

    # 3) Extracted balloons (RGBA).
    balloon_paths = []
    for idx, balloon in enumerate(balloons):
        balloon_path = balloons_dir / f"balloon_{idx:03d}.png"
        cv2.imwrite(str(balloon_path), balloon["image"])
        balloon_paths.append(str(balloon_path))

    balloon_mask_path = balloons_dir / "balloon_mask.png"
    cv2.imwrite(str(balloon_mask_path), balloon_mask)

    result = {
        "input": str(image_path),
        "output_dir": str(image_out_dir),
        "inpainted_image": str(inpainted_path),
        "panel_segmentation_image": str(panel_overlay_path),
        "num_panels": len(masks),
        "num_balloons": len(balloons),
        "balloon_images": balloon_paths,
        "balloon_mask": str(balloon_mask_path),
    }

    meta_path = image_out_dir / "result.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images: List[Path] = []
    for ext in exts:
        images.extend(sorted(input_path.glob(ext)))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panel instance segmentation + balloon extraction/inpainting pipeline"
    )
    parser.add_argument("--input", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, default="./segmentation_output", help="Output directory")

    parser.add_argument("--panel-model", type=str, default=DEFAULT_PANEL_MODEL_PATH, help="Panel model weights")
    parser.add_argument(
        "--panel-model-type",
        type=str,
        default=DEFAULT_PANEL_MODEL_TYPE,
        choices=["maskrcnn", "mask2former"],
        help="Panel model type",
    )
    parser.add_argument("--balloon-model", type=str, default=DEFAULT_BALLOON_MODEL_PATH, help="Balloon U-Net weights")
    parser.add_argument(
        "--balloon-img-size",
        type=int,
        nargs=2,
        default=list(DEFAULT_BALLOON_IMG_SIZE),
        metavar=("H", "W"),
        help="Balloon segmentation model input size",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=list(DEFAULT_IMG_SIZE),
        metavar=("H", "W"),
        help="Panel segmentation model input size",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default=DEFAULT_INPUT_TYPE,
        choices=["gray", "3ch"],
        help="Input channels for panel model",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Score threshold for panel detection",
    )
    parser.add_argument(
        "--balloon-dilate",
        type=int,
        default=2,
        help="Balloon mask dilation size before extraction",
    )
    parser.add_argument("--smooth-mask", action="store_true", help="Enable panel mask smoothing")
    parser.add_argument(
        "--smooth-kernel-size",
        type=int,
        default=5,
        help="Kernel size for panel mask smoothing",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    converter = MangaPage2Vertical(
        panel_model_path=args.panel_model,
        panel_model_type=args.panel_model_type,
        balloon_model_path=args.balloon_model,
        balloon_img_size=tuple(args.balloon_img_size),
        input_type=args.input_type,
        img_size=tuple(args.img_size),
        score_threshold=args.score_threshold,
        smooth_mask=args.smooth_mask,
        smooth_kernel_size=args.smooth_kernel_size,
        balloon_dilate=args.balloon_dilate,
    )

    results = []
    for image_path in images:
        try:
            print(f"Processing: {image_path.name}")
            result = process_single_image(converter, image_path, output_path)
            results.append(result)
            print(
                f"  saved: inpainted={result['inpainted_image']}, "
                f"panels={result['panel_segmentation_image']}, balloons={result['num_balloons']}"
            )
        except Exception as exc:
            print(f"  failed: {image_path.name}: {exc}")

    summary_path = output_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Processed {len(results)} / {len(images)} image(s).")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
