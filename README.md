# page2vertical_manga 引き継ぎ README

このリポジトリは、漫画ページ画像に対して以下を行う Python プロジェクトです。

- コマのインスタンスセグメンテーション
- コマ順序推定
- 吹き出し抽出
- 吹き出し領域の除去補間（inpainting）
- （main.py）縦読みレイアウト画像の生成

引き継ぎ向けに、実行方法・必要ファイル・チェックポイント配置をこの README に集約します。

## 1. 主要スクリプト

- `main.py`
  - 見開き/単ページを処理して、縦読み画像・比較画像・吹き出し画像などを出力。
- `segment_panels_and_balloons.py`
  - 吹き出し除去補間画像、コマセグメンテーション結果画像、抽出吹き出し画像を分けて保存する専用パイプライン。
- `predict.py`
  - コマのインスタンスセグメンテーション推論のみ（オーバーレイ保存）。
- `balloon_editor.py`
  - Streamlit UI。抽出した吹き出しを縦読み画像へ手動配置。
- `train_maskrcnn.py`
  - Mask R-CNN の学習。
- `test_mask2former.py`, `test_mask2former_gray.py`, `test_unet.py`, `test_order_estimator.py`
  - モデル評価/順序推定テスト。

## 2. 実行環境

推奨:

- Python 3.10 以上
- PyTorch + torchvision
- `transformers`
- `opencv-python`
- `numpy`, `Pillow`, `scipy`, `tqdm`

最低限のインストール例:

```bash
pip install torch torchvision
pip install transformers opencv-python numpy pillow scipy tqdm
pip install streamlit streamlit-image-coordinates
```

`main.py` と `segment_panels_and_balloons.py` は `simple_lama_inpainting` を利用します。
このリポジトリには `simple_lama_inpainting/` が含まれており、初回実行時に LAMA 重み（`big-lama.pt`）をダウンロードします。

環境変数で固定したい場合:

- `LAMA_MODEL=/absolute/path/to/big-lama.pt`
- `LAMA_MODEL_URL=<download-url>`

## 3. チェックポイント配置（重要）

### 3.1 コマセグメンテーション用（`instance_models/`）

現在、リポジトリ内に以下の学習済み重みが存在します。

- `instance_models/mask2former_gray_20251201_121013/mask2former_gray_best.pt`
- `instance_models/mask2former_3ch_20251201_121013/mask2former_best.pt`
- `instance_models/maskrcnn_gray_20251201_121013/maskrcnn_gray_best.pt`
- `instance_models/maskrcnn_3ch_20251201_121013/maskrcnn_3ch_best.pt`

注意:

- `main.py` と `segment_panels_and_balloons.py` のデフォルトは `./instance_models/mask2former_gray_best.pt` です。
- 実ファイルはサブディレクトリ内にあるため、以下のどちらかが必要です。

方法A: 実行時に明示指定

```bash
--panel-model ./instance_models/mask2former_gray_20251201_121013/mask2former_gray_best.pt
```

方法B: デフォルト名でシンボリックリンクを作成

```bash
ln -s ./mask2former_gray_20251201_121013/mask2former_gray_best.pt ./instance_models/mask2former_gray_best.pt
```

### 3.2 吹き出しセグメンテーション用（`balloon_models/`）

- `main.py` と `segment_panels_and_balloons.py` のデフォルトは
  `./balloon_models/real3000_dataset-unet-01.pt`。
- このワークスペースでは `balloon_models/` ディレクトリが未配置です。

対応:

- `balloon_models/real3000_dataset-unet-01.pt` を配置する。
- もしくは `--balloon-model /path/to/your_unet_checkpoint.pt` で指定する。

モデル未指定またはロード失敗時は、`main.py` 内の簡易白色領域検出へフォールバックします（精度は低下）。

## 4. 使い方

## 4.1 吹き出し除去補間 + コマ分割 + 吹き出し抽出（推奨）

`segment_panels_and_balloons.py` を使います。

```bash
python segment_panels_and_balloons.py \
  --input ./test_images \
  --output ./segmentation_output \
  --panel-model ./instance_models/mask2former_gray_20251201_121013/mask2former_gray_best.pt \
  --panel-model-type mask2former \
  --input-type gray \
  --balloon-model ./balloon_models/real3000_dataset-unet-01.pt
```

主な出力（画像ごと）:

```text
segmentation_output/
  <image_stem>/
    inpainted/no_balloons.png
    panel_segmentation/panels_overlay.png
    panel_segmentation/panel_mask_000.png ...
    balloons/balloon_000.png ...
    balloons/balloon_mask.png
    result.json
```

全体サマリ:

- `segmentation_output/summary.json`

## 4.2 見開きを縦読みへ変換

`main.py` を使います。

```bash
python main.py \
  --input ./test_images \
  --output ./output_vertical \
  --panel-model ./instance_models/mask2former_gray_20251201_121013/mask2former_gray_best.pt \
  --panel-model-type mask2former \
  --input-type gray \
  --balloon-model ./balloon_models/real3000_dataset-unet-01.pt \
  --score-threshold 0.5
```

主な出力（画像ごと）:

```text
output_vertical/
  <image_stem>/
    original.<ext>
    vertical.png
    comparison_page.png (または right/left)
    panels/panel_000.png ...
    balloons/balloon_000.png ...
    masks/mask_000.png ...
    meta.json
```

全体サマリ:

- `output_vertical/conversion_summary.json`

## 4.3 セグメンテーション推論のみ

`predict.py` を使います。

```bash
python predict.py \
  --model mask2former \
  --weights ./instance_models/mask2former_gray_20251201_121013/mask2former_gray_best.pt \
  --input ./test_images \
  --input-type gray \
  --output ./predictions
```

## 4.4 吹き出し手動配置 UI

```bash
streamlit run balloon_editor.py
```

デフォルトでは `./output_m2f` を見に行くため、必要に応じて UI 側で出力ディレクトリを変更してください。

## 5. 学習・評価のデータ形式

### 5.1 Mask R-CNN 学習（`train_maskrcnn.py`）

想定ルート:

- `--root <dataset_root>`
- `train/`, `val/` があれば split として使用。なければ `<dataset_root>` 直下を使用。

必要構造（例）:

```text
<dataset_root>/
  train/
    images/*.jpg|png
    instance_masks/*_instance.png
    lsd/*_lsd.png          # input-type=3ch の場合
    sdf/*_sdf.png          # input-type=3ch の場合
  val/
    ... 同様 ...
```

実行例:

```bash
python train_maskrcnn.py \
  --root ./frame_dataset/1000_instance \
  --input-type gray \
  --epochs 50 \
  --batch 4 \
  --output ./panel_models/maskrcnn_gray
```

### 5.2 U-Net 評価（`test_unet.py`）

必要構造:

```text
<data-root>/
  images/*.jpg|png
  masks/*_mask.png
```

実行例:

```bash
python test_unet.py \
  --model-path ./balloon_models/real3000_dataset-unet-01.pt \
  --data-root ./test_dataset \
  --result-dir ./test_results
```

## 6. 既知の注意点

- `main.py`/`segment_panels_and_balloons.py` のデフォルト `--panel-model` は、現状配置の実ファイルパスと異なります。
  - 実行時に `--panel-model` を明示指定するのが安全です。
- `balloon_models/` はこのワークスペースに存在しないため、別途配置が必要です。
- `test_mask2former.py` と `test_mask2former_gray.py` は
  `utils.instance_metrics` を import しますが、現状トップレベル `utils/` は見当たりません。
  - これら評価スクリプトを使う場合は、該当モジュールを追加するか import を修正してください。

## 7. クイックスタート（最短）

1. コマモデルを指定して、吹き出し抽出・補間パイプラインを実行。

```bash
python segment_panels_and_balloons.py \
  --input ./test_images \
  --output ./segmentation_output \
  --panel-model ./instance_models/mask2former_gray_20251201_121013/mask2former_gray_best.pt \
  --panel-model-type mask2former \
  --input-type gray \
  --balloon-model ./balloon_models/real3000_dataset-unet-01.pt
```

2. 結果確認:

- 吹き出し除去補間画像: `segmentation_output/<stem>/inpainted/no_balloons.png`
- コマ分割結果: `segmentation_output/<stem>/panel_segmentation/panels_overlay.png`
- 抽出吹き出し: `segmentation_output/<stem>/balloons/balloon_*.png`

## 8. トラブルシュート

- `FileNotFoundError: ...mask2former_gray_best.pt` が出る場合:
  - `--panel-model` にサブディレクトリ内の実ファイルを指定してください。
- `balloon model ... not found` が出る場合:
  - `--balloon-model` で有効な `.pt` を指定するか、`balloon_models/` に配置してください。
- LAMA の初回実行で遅い場合:
  - モデルダウンロードが走っています。2回目以降はキャッシュが使われます。