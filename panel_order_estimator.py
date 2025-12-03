from copy import deepcopy

def panel_order_estimater(panels, img_width, img_height):
    """
    コマ順序を推定する（擬似的なコマ領域を用いた齋藤らの手順の再現版）
    Args:
        panels (list of dict): コマのバウンディングボックス情報
            [{'type': 'frame', 'id': '...', 'xmin': '...', 'ymin': '...', 'xmax': '...', 'ymax': '...'}, ...]
        img_width (int): ページ画像の幅
        img_height (int): ページ画像の高さ
    Returns:
        list: 順序付けされたコマの情報リスト（元のパネル dict に擬似領域情報が付いたもの）
    """

    # ---- 1. バウンディングボックスの座標を数値に変換 & 擬似コマ領域の初期化 ----
    panels = deepcopy(panels)  # 元データを壊さない
    for panel in panels:
        panel['xmin'] = int(panel['xmin'])
        panel['ymin'] = int(panel['ymin'])
        panel['xmax'] = int(panel['xmax'])
        panel['ymax'] = int(panel['ymax'])
        panel['center_x'] = (panel['xmin'] + panel['xmax']) // 2
        panel['center_y'] = (panel['ymin'] + panel['ymax']) // 2

        # 擬似的なコマ領域の初期値（最初は元の bbox と同じ）
        panel['pxmin'] = float(panel['xmin'])
        panel['pymin'] = float(panel['ymin'])
        panel['pxmax'] = float(panel['xmax'])
        panel['pymax'] = float(panel['ymax'])

    # ---- 2. バウンディングボックスが重なっている場合に擬似コマ領域を作る ----
    def resolve_overlaps(panels):
        """
        バウンディングボックスが重なっている場合，
        重なった領域の矩形の中央線で上下または左右に分割して，
        互いに重ならない擬似的なコマ領域(pxmin, pymin, pxmax, pymax)を作る
        """
        max_iter = 10
        n = len(panels)
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                a = panels[i]
                for j in range(i + 1, n):
                    b = panels[j]

                    # 交差矩形を計算
                    ixmin = max(a['pxmin'], b['pxmin'])
                    iymin = max(a['pymin'], b['pymin'])
                    ixmax = min(a['pxmax'], b['pxmax'])
                    iymax = min(a['pymax'], b['pymax'])
                    if ixmin >= ixmax or iymin >= iymax:
                        continue  # 重なりなし

                    # 交差あり → 交差領域の中央線で分割
                    # どちらが左右/上下かはコマの中心位置から判定する
                    cx_a = (a['pxmin'] + a['pxmax']) / 2.0
                    cy_a = (a['pymin'] + a['pymax']) / 2.0
                    cx_b = (b['pxmin'] + b['pxmax']) / 2.0
                    cy_b = (b['pymin'] + b['pymax']) / 2.0

                    # 中心の差が横方向に大きければ左右関係，縦方向に大きければ上下関係
                    if abs(cx_a - cx_b) >= abs(cy_a - cy_b):
                        # 左右に並んでいるとみなして，縦の中央線で分割
                        xmid = (ixmin + ixmax) / 2.0
                        if cx_a <= cx_b:
                            left, right = a, b
                        else:
                            left, right = b, a

                        # left の右端を xmid まで削る
                        new_left_pxmax = min(left['pxmax'], xmid)
                        if left['pxmin'] < new_left_pxmax < left['pxmax']:
                            left['pxmax'] = new_left_pxmax
                            changed = True

                        # right の左端を xmid まで削る
                        new_right_pxmin = max(right['pxmin'], xmid)
                        if right['pxmin'] < new_right_pxmin < right['pxmax']:
                            right['pxmin'] = new_right_pxmin
                            changed = True
                    else:
                        # 上下に並んでいるとみなして，横の中央線で分割
                        ymid = (iymin + iymax) / 2.0
                        if cy_a <= cy_b:
                            top, bottom = a, b
                        else:
                            top, bottom = b, a

                        # top の下端を ymid まで削る
                        new_top_pymax = min(top['pymax'], ymid)
                        if top['pymin'] < new_top_pymax < top['pymax']:
                            top['pymax'] = new_top_pymax
                            changed = True

                        # bottom の上端を ymid まで削る
                        new_bottom_pymin = max(bottom['pymin'], ymid)
                        if bottom['pymin'] < new_bottom_pymin < bottom['pymax']:
                            bottom['pymin'] = new_bottom_pymin
                            changed = True

            if not changed:
                break

    resolve_overlaps(panels)

    # ---- 3. 手順 1〜4 による順序付け ----

    ordered_panels = []          # 読み順に並べたコマ
    undefined_panels = panels[:] # まだ順序が未定義のコマ

    def v_overlap(a, b):
        """縦方向のオーバーラップ判定（擬似コマ領域で判定）"""
        return not (a['pymax'] <= b['pymin'] or a['pymin'] >= b['pymax'])

    # 手順1で使う: 「上側に順序が未定義のコマがない」かどうか
    def has_panel_above(panel, candidates):
        for other in candidates:
            if other is panel:
                continue
            # 横方向にオーバーラップしている場合だけ「上下関係」を見る
            if other['pxmax'] <= panel['pxmin'] or other['pxmin'] >= panel['pxmax']:
                continue
            # other が panel より「上」にある
            if other['pymax'] <= panel['pymin']:
                return True
        return False

    def find_closest_to_top_right(candidates, current_bottom):
        """
        手順1:
          上側に未定義コマがない集合の中から,
          擬似コマ領域の右上座標が「切り取り後ページ」の右上に最も近いコマを選ぶ
        """
        if not candidates:
            return None
        top_right_x = img_width
        top_right_y = current_bottom  # ページを current_bottom で切り取ったとみなす
        closest_panel = None
        min_distance = float('inf')
        for panel in candidates:
            # 擬似コマ領域の右上
            x = panel['pxmax']
            y = panel['pymin']
            distance = (top_right_x - x) ** 2 + (top_right_y - y) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_panel = panel
        return closest_panel

    def is_leftmost(panel, all_panels):
        """
        手順2:
          「左端」はページの左端ではなく，
          『縦方向にオーバーラップする範囲で，自分の左側にコマが存在しないこと』
          と解釈して実装
        """
        for other in all_panels:
            if other is panel:
                continue
            # 同じ「段」に属しそうなもの（縦方向にオーバーラップ）
            if not v_overlap(other, panel):
                continue
            # 擬似コマ領域の左辺より左側に別コマがある
            if other['pxmax'] <= panel['pxmin']:
                return False
        return True

    def find_available_panels(candidates):
        """
        手順3:
          擬似的なコマ領域の下辺より上側かつ左辺より右側に，
          順序が未定義のコマが存在しないコマの集合 S を作る
        """
        if len(candidates) <= 1:
            return candidates[:]
        available_panels = []
        for panel in candidates:
            is_available = True
            for other in candidates:
                if other is panel:
                    continue
                # other['pymax'] < panel['pymax'] → panel より「上側」にある
                # other['pxmax'] > panel['pxmin'] → panel の左辺より右側に領域がある
                if other['pymax'] < panel['pymax'] and other['pxmax'] > panel['pxmin']:
                    is_available = False
                    break
            if is_available:
                available_panels.append(panel)
        return available_panels

    def find_nearest_panel(panel, candidates):
        """
        手順3:
          直前のコマの左下座標に最も近いコマを，
          候補集合の中から選ぶ（擬似コマ領域の頂点を使用）
        """
        if not candidates:
            return None
        nearest_panel = None
        bottom_left_x = panel['pxmin']
        bottom_left_y = panel['pymax']
        min_distance = float('inf')
        for other in candidates:
            # other の擬似コマ領域の4頂点との距離のうち最小のものを採用
            xs = (other['pxmin'], other['pxmax'])
            ys = (other['pymin'], other['pymax'])
            distances = [
                (bottom_left_x - xs[0]) ** 2 + (bottom_left_y - ys[0]) ** 2,
                (bottom_left_x - xs[1]) ** 2 + (bottom_left_y - ys[0]) ** 2,
                (bottom_left_x - xs[0]) ** 2 + (bottom_left_y - ys[1]) ** 2,
                (bottom_left_x - xs[1]) ** 2 + (bottom_left_y - ys[1]) ** 2,
            ]
            d = min(distances)
            if d < min_distance:
                min_distance = d
                nearest_panel = other
        return nearest_panel

    current_bottom = 0.0  # 「ここでページを切る」とみなす y 座標

    while undefined_panels:
        # --- 手順1: ブロックの最初のコマを選ぶ ---
        # 上側に順序が未定義のコマがないものだけを候補にする
        top_candidates = [p for p in undefined_panels if not has_panel_above(p, undefined_panels)]
        if not top_candidates:
            # 何らかの理由で上側条件を満たすものがない場合は，未定義コマ全体から選ぶ
            top_candidates = undefined_panels[:]

        next_panel = find_closest_to_top_right(top_candidates, current_bottom)
        ordered_panels.append(next_panel)
        undefined_panels.remove(next_panel)

        # --- 同じ「段」の左方向へ進んでいく部分 (手順2,3) ---
        while True:
            # 手順2: 「左側にコマがない」なら左端 → ここでページを切って次のブロックへ
            if is_leftmost(next_panel, panels):
                current_bottom = next_panel['pymax']
                break

            # 手順3: 利用可能集合 S を作り，その中から次のコマを選ぶ
            candidates = find_available_panels(undefined_panels)
            if not candidates:
                # 利用可能な候補が無ければ，この段はここで終わり
                current_bottom = next_panel['pymax']
                break

            new_panel = find_nearest_panel(next_panel, candidates)
            if new_panel is None:
                current_bottom = next_panel['pymax']
                break

            ordered_panels.append(new_panel)
            undefined_panels.remove(new_panel)
            next_panel = new_panel

    return ordered_panels


if __name__ == '__main__':
    # テストデータ
    data = [
        {'type': 'frame', 'id': '00000009', 'xmin': '899', 'ymin': '585', 'xmax': '1170', 'ymax': '1085'},
        {'type': 'frame', 'id': '0000000c', 'xmin': '2',   'ymin': '0',   'xmax': '826',  'ymax': '513'},
        {'type': 'frame', 'id': '0000000e', 'xmin': '72',  'ymin': '516', 'xmax': '743',  'ymax': '1101'},
        {'type': 'frame', 'id': '00000014', 'xmin': '906', 'ymin': '95',  'xmax': '1575', 'ymax': '576'},
        {'type': 'frame', 'id': '0000001d', 'xmin': '1167','ymin': '588', 'xmax': '1580', 'ymax': '1090'}
    ]
    img_w = max(int(p['xmax']) for p in data)
    img_h = max(int(p['ymax']) for p in data)

    import pprint
    pprint.pp(panel_order_estimater(data, img_w, img_h))
