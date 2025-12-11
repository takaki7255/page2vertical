from copy import deepcopy
from typing import List, Dict, Any


def panel_order_estimater_single_page(panels: List[Dict[str, Any]], img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    単一ページ（見開きではない）のコマ順序推定
    
    見開きページを既に分割済みの場合や、単一ページの漫画に使用する。
    ページ全体を右ページとして扱い、右上から左下への順序で並べる。
    
    Args:
        panels: コマ情報のリスト
        img_width: 画像幅
        img_height: 画像高さ
    
    Returns:
        ordered_panels: 順序付けされたコマリスト
    """
    # 深いコピーを取って元データを壊さない
    panels = deepcopy(panels)
    
    if not panels:
        return []
    
    # 数値変換 & 中心座標
    for p in panels:
        p["xmin"] = int(p["xmin"])
        p["ymin"] = int(p["ymin"])
        p["xmax"] = int(p["xmax"])
        p["ymax"] = int(p["ymax"])
        p["center_x"] = (p["xmin"] + p["xmax"]) // 2
        p["center_y"] = (p["ymin"] + p["ymax"]) // 2
        # 擬似コマ領域はそのまま元の領域を使用
        p["pxmin"] = float(p["xmin"])
        p["pymin"] = float(p["ymin"])
        p["pxmax"] = float(p["xmax"])
        p["pymax"] = float(p["ymax"])
    
    # 重なり解消
    _resolve_overlaps(panels)
    
    # 順序推定（ページ全体を右ページとして扱う）
    ordered = _order_one_page(panels, anchor_right=img_width, all_panels=panels)
    
    return ordered


def _resolve_overlaps(page_panels: List[Dict[str, Any]]) -> None:
    """
    擬似コマ領域同士が重なっている場合，
    重なった領域の中央線で上下 or 左右に分割し，
    最終的に（近似的に）互いに重ならないような擬似コマ領域にする。
    """
    max_iter = 10
    n = len(page_panels)
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            a = page_panels[i]
            for j in range(i + 1, n):
                b = page_panels[j]

                ixmin = max(a["pxmin"], b["pxmin"])
                iymin = max(a["pymin"], b["pymin"])
                ixmax = min(a["pxmax"], b["pxmax"])
                iymax = min(a["pymax"], b["pymax"])

                if ixmin >= ixmax or iymin >= iymax:
                    continue  # 重なりなし

                # 交差領域の中心
                cx_a = (a["pxmin"] + a["pxmax"]) / 2.0
                cy_a = (a["pymin"] + a["pymax"]) / 2.0
                cx_b = (b["pxmin"] + b["pxmax"]) / 2.0
                cy_b = (b["pymin"] + b["pymax"]) / 2.0

                # 横方向の差が大きい → 左右に並んでいるとみなして垂直線で分割
                # 縦方向の差が大きい → 上下に並んでいるとみなして水平線で分割
                if abs(cx_a - cx_b) >= abs(cy_a - cy_b):
                    # 左右方向の分割
                    xmid = (ixmin + ixmax) / 2.0
                    if cx_a <= cx_b:
                        left, right = a, b
                    else:
                        left, right = b, a

                    new_left_pxmax = min(left["pxmax"], xmid)
                    if left["pxmin"] < new_left_pxmax < left["pxmax"]:
                        left["pxmax"] = new_left_pxmax
                        changed = True

                    new_right_pxmin = max(right["pxmin"], xmid)
                    if right["pxmin"] < new_right_pxmin < right["pxmax"]:
                        right["pxmin"] = new_right_pxmin
                        changed = True
                else:
                    # 上下方向の分割
                    ymid = (iymin + iymax) / 2.0
                    if cy_a <= cy_b:
                        top, bottom = a, b
                    else:
                        top, bottom = b, a

                    new_top_pymax = min(top["pymax"], ymid)
                    if top["pymin"] < new_top_pymax < top["pymax"]:
                        top["pymax"] = new_top_pymax
                        changed = True

                    new_bottom_pymin = max(bottom["pymin"], ymid)
                    if bottom["pymin"] < new_bottom_pymin < bottom["pymax"]:
                        bottom["pymin"] = new_bottom_pymin
                        changed = True

        if not changed:
            break


def _order_one_page(page_panels: List[Dict[str, Any]],
                    anchor_right: float,
                    all_panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    1ページ分のコマだけを受け取り，論文の手順で読み順を決定する。
    """
    if not page_panels:
        return []

    ordered: List[Dict[str, Any]] = []
    undefined: List[Dict[str, Any]] = page_panels.copy()

    # 手順1用: 「上側に順序が未定義のコマがあるか？」
    def has_panel_above(panel: Dict[str, Any],
                        candidates: List[Dict[str, Any]]) -> bool:
        for other in candidates:
            if other is panel:
                continue
            if other["pymax"] <= panel["pymin"]:
                return True
        return False

    # 手順1: アンカーの右上に最も近いコマを探す
    def find_closest_to_top_right(cands: List[Dict[str, Any]],
                                  current_bottom: float) -> Dict[str, Any]:
        tx = anchor_right
        ty = current_bottom
        best = None
        best_dist = float("inf")
        for p in cands:
            # 擬似コマ領域の右上座標
            x = p["pxmax"]
            y = p["pymin"]
            d = (tx - x) ** 2 + (ty - y) ** 2
            if d < best_dist:
                best_dist = d
                best = p
        return best

    # 手順2: 「左端」判定
    def is_leftmost(panel: Dict[str, Any],
                    check_panels: List[Dict[str, Any]]) -> bool:
        for other in check_panels:
            if other is panel:
                continue
            if other["pxmax"] <= panel["pxmin"]:
                return False
        return True

    # 手順3: 利用可能なコマ集合S
    def find_available_panels(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(cands) <= 1:
            return cands[:]

        available: List[Dict[str, Any]] = []
        for p in cands:
            ok = True
            for other in cands:
                if other is p:
                    continue
                if other["pymax"] < p["pymax"] and other["pxmax"] > p["pxmin"]:
                    ok = False
                    break
            if ok:
                available.append(p)
        return available

    # 手順3: 左下に最も近いコマ
    def find_nearest_panel(panel: Dict[str, Any],
                           cands: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not cands:
            return None
        bx = panel["pxmin"]
        by = panel["pymax"]
        best = None
        best_dist = float("inf")
        for other in cands:
            xs = (other["pxmin"], other["pxmax"])
            ys = (other["pymin"], other["pymax"])
            d = min(
                (bx - xs[0]) ** 2 + (by - ys[0]) ** 2,
                (bx - xs[1]) ** 2 + (by - ys[0]) ** 2,
                (bx - xs[0]) ** 2 + (by - ys[1]) ** 2,
                (bx - xs[1]) ** 2 + (by - ys[1]) ** 2,
            )
            if d < best_dist:
                best_dist = d
                best = other
        return best

    current_bottom = 0.0

    while undefined:
        # 手順1: 上に未定義コマがない集合から右上に最も近いコマを選ぶ
        top_candidates = [p for p in undefined if not has_panel_above(p, undefined)]
        if not top_candidates:
            top_candidates = undefined[:]

        first = find_closest_to_top_right(top_candidates, current_bottom)
        ordered.append(first)
        undefined.remove(first)
        current = first

        # 手順2,3: 同一「段」で左方向へ辿る
        while True:
            # 手順2: 左端ならここでブロックを切る
            if is_leftmost(current, all_panels):
                current_bottom = current["pymax"]
                break

            # 手順3: 利用可能集合Sから左下に最も近いコマ
            cand = find_available_panels(undefined)
            if not cand:
                current_bottom = current["pymax"]
                break

            nxt = find_nearest_panel(current, cand)
            if nxt is None:
                current_bottom = current["pymax"]
                break

            ordered.append(nxt)
            undefined.remove(nxt)
            current = nxt

    return ordered


def panel_order_estimater(panels: List[Dict[str, Any]], img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    見開きページのコマ順序推定（齋藤らの手法をベースに実装）

    - まずノド位置で左右ページに分割
    - ノドを跨ぐコマは，順序判定に使う擬似コマ領域だけノドでクリップ
    - 各ページごとに
        - コマ重なりから擬似コマ領域を生成（中央線で分割）
        - 「上にコマがない」「左端」「利用可能集合S」「左下からの最近傍」
          を用いて読み順を決める
    - 日本の単行本を想定して「右ページ → 左ページ」の順に連結して返す
    """

    # 深いコピーを取って元データを壊さない
    panels = deepcopy(panels)

    # 数値変換 & 中心座標
    for p in panels:
        p["xmin"] = int(p["xmin"])
        p["ymin"] = int(p["ymin"])
        p["xmax"] = int(p["xmax"])
        p["ymax"] = int(p["ymax"])
        p["center_x"] = (p["xmin"] + p["xmax"]) // 2
        p["center_y"] = (p["ymin"] + p["ymax"]) // 2

    gutter_x = img_width / 2.0  # ノド位置（簡易には画像中央）

    # --- 1. ノドで左右ページに分割しつつ，ノド跨ぎコマは擬似領域だけクリップ ---
    right_panels: List[Dict[str, Any]] = []
    left_panels: List[Dict[str, Any]] = []

    for p in panels:
        xmin, xmax = p["xmin"], p["xmax"]
        ymin, ymax = p["ymin"], p["ymax"]
        cx = p["center_x"]

        # ページ（right/left）と，そのページ内で使う擬似コマ領域 pxmin/pxmax を決める
        if xmax <= gutter_x:
            # 完全に左ページ
            page = "left"
            pxmin, pxmax = xmin, xmax
        elif xmin >= gutter_x:
            # 完全に右ページ
            page = "right"
            pxmin, pxmax = xmin, xmax
        else:
            # ノドを跨ぐコマ：中心位置側のページとして扱い，片側だけを擬似領域として使う
            if cx >= gutter_x:
                page = "right"
                pxmin = gutter_x
                pxmax = xmax
            else:
                page = "left"
                pxmin = xmin
                pxmax = gutter_x

        # 擬似コマ領域（後で重なり解消 & 順序判定に使用）
        p["pxmin"] = float(pxmin)
        p["pymin"] = float(ymin)
        p["pxmax"] = float(pxmax)
        p["pymax"] = float(ymax)

        if page == "right":
            right_panels.append(p)
        else:
            left_panels.append(p)

    # --- 2. 擬似コマ領域の重なりを解消する（同一ページ内） ---
    def resolve_overlaps(page_panels: List[Dict[str, Any]]) -> None:
        """
        擬似コマ領域同士が重なっている場合，
        重なった領域の中央線で上下 or 左右に分割し，
        最終的に（近似的に）互いに重ならないような擬似コマ領域にする。
        """
        max_iter = 10
        n = len(page_panels)
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                a = page_panels[i]
                for j in range(i + 1, n):
                    b = page_panels[j]

                    ixmin = max(a["pxmin"], b["pxmin"])
                    iymin = max(a["pymin"], b["pymin"])
                    ixmax = min(a["pxmax"], b["pxmax"])
                    iymax = min(a["pymax"], b["pymax"])

                    if ixmin >= ixmax or iymin >= iymax:
                        continue  # 重なりなし

                    # 交差領域の中心
                    cx_a = (a["pxmin"] + a["pxmax"]) / 2.0
                    cy_a = (a["pymin"] + a["pymax"]) / 2.0
                    cx_b = (b["pxmin"] + b["pxmax"]) / 2.0
                    cy_b = (b["pymin"] + b["pymax"]) / 2.0

                    # 横方向の差が大きい → 左右に並んでいるとみなして垂直線で分割
                    # 縦方向の差が大きい → 上下に並んでいるとみなして水平線で分割
                    if abs(cx_a - cx_b) >= abs(cy_a - cy_b):
                        # 左右方向の分割
                        xmid = (ixmin + ixmax) / 2.0
                        if cx_a <= cx_b:
                            left, right = a, b
                        else:
                            left, right = b, a

                        new_left_pxmax = min(left["pxmax"], xmid)
                        if left["pxmin"] < new_left_pxmax < left["pxmax"]:
                            left["pxmax"] = new_left_pxmax
                            changed = True

                        new_right_pxmin = max(right["pxmin"], xmid)
                        if right["pxmin"] < new_right_pxmin < right["pxmax"]:
                            right["pxmin"] = new_right_pxmin
                            changed = True
                    else:
                        # 上下方向の分割
                        ymid = (iymin + iymax) / 2.0
                        if cy_a <= cy_b:
                            top, bottom = a, b
                        else:
                            top, bottom = b, a

                        new_top_pymax = min(top["pymax"], ymid)
                        if top["pymin"] < new_top_pymax < top["pymax"]:
                            top["pymax"] = new_top_pymax
                            changed = True

                        new_bottom_pymin = max(bottom["pymin"], ymid)
                        if bottom["pymin"] < new_bottom_pymin < bottom["pymax"]:
                            bottom["pymin"] = new_bottom_pymin
                            changed = True

            if not changed:
                break

    # --- 3. 1ページ分の順序付け（論文の手順 1〜4） ---
    def order_one_page(page_panels: List[Dict[str, Any]],
                       anchor_right: float) -> List[Dict[str, Any]]:
        """
        1ページ分のコマだけを受け取り，論文の手順で読み順を決定する。
        anchor_right:
            右ページ → img_width
            左ページ → gutter_x
            として渡し，そのページ内の「右上」に相当するアンカーとする。
        """
        if not page_panels:
            return []

        # 重なり解消（擬似コマ領域）
        resolve_overlaps(page_panels)

        ordered: List[Dict[str, Any]] = []
        undefined: List[Dict[str, Any]] = page_panels.copy()

        # 手順1用: 「上側に順序が未定義のコマがあるか？」
        # → 擬似コマ領域の上辺 pymin より上に，他コマの下辺 pymax があるかどうか
        def has_panel_above(panel: Dict[str, Any],
                            candidates: List[Dict[str, Any]]) -> bool:
            for other in candidates:
                if other is panel:
                    continue
                if other["pymax"] <= panel["pymin"]:
                    return True
            return False

        # 手順1: アンカーの右上に最も近いコマを探す
        def find_closest_to_top_right(cands: List[Dict[str, Any]],
                                      current_bottom: float) -> Dict[str, Any]:
            tx = anchor_right
            ty = current_bottom
            best = None
            best_dist = float("inf")
            for p in cands:
                # 擬似コマ領域の右上座標
                x = p["pxmax"]
                y = p["pymin"]
                d = (tx - x) ** 2 + (ty - y) ** 2
                if d < best_dist:
                    best_dist = d
                    best = p
            return best

        # 手順2: 「左端」判定
        # 論文の「横軸左側に別のコマが存在しない」を
        # 「同一ページ内で，自分より完全に左側にあるコマがない」として実装
        def is_leftmost(panel: Dict[str, Any],
                        all_panels: List[Dict[str, Any]]) -> bool:
            for other in all_panels:
                if other is panel:
                    continue
                if other["pxmax"] <= panel["pxmin"]:
                    return False
            return True

        # 手順3: 利用可能なコマ集合S
        def find_available_panels(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            擬似コマ領域の下辺より上側かつ左辺より右側に，
            未定義コマが存在しないコマの集合Sを返す。
            （論文本文の条件をそのまま擬似領域で実装）
            """
            if len(cands) <= 1:
                return cands[:]

            available: List[Dict[str, Any]] = []
            for p in cands:
                ok = True
                for other in cands:
                    if other is p:
                        continue
                    if other["pymax"] < p["pymax"] and other["pxmax"] > p["pxmin"]:
                        ok = False
                        break
                if ok:
                    available.append(p)
            return available

        # 手順3: 左下に最も近いコマ
        def find_nearest_panel(panel: Dict[str, Any],
                               cands: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not cands:
                return None
            bx = panel["pxmin"]
            by = panel["pymax"]
            best = None
            best_dist = float("inf")
            for other in cands:
                xs = (other["pxmin"], other["pxmax"])
                ys = (other["pymin"], other["pymax"])
                d = min(
                    (bx - xs[0]) ** 2 + (by - ys[0]) ** 2,
                    (bx - xs[1]) ** 2 + (by - ys[0]) ** 2,
                    (bx - xs[0]) ** 2 + (by - ys[1]) ** 2,
                    (bx - xs[1]) ** 2 + (by - ys[1]) ** 2,
                )
                if d < best_dist:
                    best_dist = d
                    best = other
            return best

        current_bottom = 0.0  # 「ここから下が未処理のブロック」とみなすy座標

        while undefined:
            # 手順1: 上に未定義コマがない集合から右上に最も近いコマを選ぶ
            top_candidates = [p for p in undefined if not has_panel_above(p, undefined)]
            if not top_candidates:
                # 何らかの理由で空になった場合は未定義からフォールバック
                top_candidates = undefined[:]

            first = find_closest_to_top_right(top_candidates, current_bottom)
            ordered.append(first)
            undefined.remove(first)
            current = first

            # 手順2,3: 同一「段」で左方向へ辿る
            while True:
                # 手順2: 左端ならここでブロックを切る
                if is_leftmost(current, page_panels):
                    current_bottom = current["pymax"]
                    break

                # 手順3: 利用可能集合Sから左下に最も近いコマ
                cand = find_available_panels(undefined)
                if not cand:
                    current_bottom = current["pymax"]
                    break

                nxt = find_nearest_panel(current, cand)
                if nxt is None:
                    current_bottom = current["pymax"]
                    break

                ordered.append(nxt)
                undefined.remove(nxt)
                current = nxt

        return ordered

    # --- 4. 右ページ → 左ページの順で結合して返す ---
    ordered_all: List[Dict[str, Any]] = []
    # 右ページ（日本語マンガは右から）
    ordered_all.extend(order_one_page(right_panels, anchor_right=img_width))
    # 左ページ
    ordered_all.extend(order_one_page(left_panels, anchor_right=gutter_x))

    return ordered_all


if __name__ == "__main__":
    # あなたが出してくれたテスト例2つをざっくり確認する用

    data1 = [
        {'type': 'frame', 'id': '00000014', 'xmin': '906',  'ymin': '95',  'xmax': '1575', 'ymax': '576'},
        {'type': 'frame', 'id': '00000009', 'xmin': '899',  'ymin': '585', 'xmax': '1170', 'ymax': '1085'},
        {'type': 'frame', 'id': '0000001d', 'xmin': '1167', 'ymin': '588', 'xmax': '1580', 'ymax': '1090'},
        {'type': 'frame', 'id': '0000000c', 'xmin': '2',    'ymin': '0',   'xmax': '826',  'ymax': '513'},
        {'type': 'frame', 'id': '0000000e', 'xmin': '72',   'ymin': '516', 'xmax': '743',  'ymax': '1101'},
    ]

    data2 = [
        {'type': 'frame', 'id': '0001289a', 'xmin': '893', 'ymin': '0',   'xmax': '1615', 'ymax': '428'},
        {'type': 'frame', 'id': '0001287e', 'xmin': '1307','ymin': '436', 'xmax': '1613', 'ymax': '1098'},
        {'type': 'frame', 'id': '00012880', 'xmin': '1467','ymin': '375', 'xmax': '1619', 'ymax': '656'},
        {'type': 'frame', 'id': '00012885', 'xmin': '1081','ymin': '436', 'xmax': '1294', 'ymax': '1098'},
        {'type': 'frame', 'id': '0001288a', 'xmin': '891', 'ymin': '433', 'xmax': '1073', 'ymax': '1092'},
        {'type': 'frame', 'id': '00012881', 'xmin': '36',  'ymin': '0',   'xmax': '751',  'ymax': '489'},
        {'type': 'frame', 'id': '00012897', 'xmin': '36',  'ymin': '390', 'xmax': '756',  'ymax': '1169'},
    ]

    # 画像サイズはざっくり最大値に合わせておく
    def img_size(ds):
        w = max(int(d["xmax"]) for d in ds)
        h = max(int(d["ymax"]) for d in ds)
        return w, h

    for name, ds in [("data1", data1), ("data2", data2)]:
        w, h = img_size(ds)
        ordered = panel_order_estimater(ds, w, h)
        print(name, "→", [p["id"] for p in ordered])
        # 期待:
        # data1 → ['00000014', '0000001d', '00000009', '0000000c', '0000000e']
        # data2 → ['0001289a', '00012880', '0001287e', '00012885', '0001288a', '00012881', '00012897']
