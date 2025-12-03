#だいぶ改善の余地あり
#バウンディングボックスが重なっている場合の処理が未実装
#ちょいちょい誤推定あり．原因わからず．無念．正推定たぶん9割5分くらいあるんじゃないかな．とりあえず今日（2024/12/19）はこの辺で．^^;
#2024/12/19 原因わかってきた．左端に来た時に右上に近いコマを選ぶと左側にある小さいコマが飛ばされる
#バウンディングボックス重なってるときの処理してないからこまが重なってるときfind_available_panels関数がうまくいかない
def panel_order_estimater(panels, img_width, img_height):
    """
    コマ順序を推定する（擬似的なコマ領域を考慮）
    Args:
        panels (list of dict): コマのバウンディングボックス情報
            [{'type': 'frame', 'id': '...', 'xmin': '...', 'ymin': '...', 'xmax': '...', 'ymax': '...'}, ...]
    Returns:
        list: 順序付けされたコマの情報リスト（座標情報含む）
    """
    # 1. バウンディングボックスの座標を数値に変換
    for panel in panels:
        panel['xmin'] = int(panel['xmin'])
        panel['ymin'] = int(panel['ymin'])
        panel['xmax'] = int(panel['xmax'])
        panel['ymax'] = int(panel['ymax'])
        panel['center_x'] = (panel['xmin'] + panel['xmax']) // 2
        panel['center_y'] = (panel['ymin'] + panel['ymax']) // 2


    # # 擬似的なバウンディングボックスを生成する関数
    # def define_non_overlapping_boxes(panels):
    #     """重なりがある場合に重ならないようにバウンディングボックスを調整"""
    #     adjusted_panels = []
    #     overlaps = []
    #     # panelsの中身が一つの場合はそのまま返す
    #     if len(panels) == 1:
    #         return panels
    #     for i, panel in enumerate(panels):
    #         for other in panels[i + 1:]:
    #             # 重なりが右側にある場合
    #             if panel['xmax'] > other['xmin'] and 

    #     return adjusted_panels

    # # 重なりを解消したパネルリストを取得
    # panels = define_non_overlapping_boxes(panels)

    # 2. 順序付けのアルゴリズム
    ordered_panels = []  # コマ順序を格納
    undefined_panels = panels.copy()  # 未定義のコマリスト

    def find_closest_to_top_right(panels, current_bottom):
        """ページ右上座標に最も近いコマを見つける (xmax が最大、ymin が最小の条件)"""
        top_right_x = img_width
        top_right_y = current_bottom
        closest_panel = None
        min_distance = float('inf')
        for panel in panels:
            #top_rightとコマ右上との角度を計算
            
            distance = (top_right_x - panel['xmax']) ** 2 + (top_right_y - panel['ymin']) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_panel = panel
        return closest_panel
        

    def is_leftmost(panel, panels):
        """コマがページの左端に位置するか確認する"""
        # panelの左下座標より左上にコマの中心がないか確認
        for other in panels:
            if other['center_x'] < panel['xmin'] and other['center_y'] < panel['ymax']:
                return False
        return True
    
    def find_available_panels(panels):
        """順序が未定義のコマの中でコマの下辺より上側かつ左辺より右 側に，順序が未定義のコマが存在しないコマを探す"""
        available_panels = []
        if len(panels) == 1:
            return panels
        for panel in panels:
            is_available = True
            for other in panels:
                if other == panel:
                    continue
                if other['ymax'] < panel['ymax'] and other['xmax'] > panel['xmin']:
                    is_available = False
                    break
            if is_available:
                available_panels.append(panel)
        return available_panels
    
    def find_nearest_panel(panel, panels):
        """指定したコマの左下座標に最も近いコマを見つける"""
        nearest_panel = None
        bottom_left_x = panel['xmin']
        bottom_left_y = panel['ymax']
        min_distance = float('inf')
        for other in panels:
            if other == panel:
                continue
            # 四つの頂点との距離のうち最小のものをdistanceに格納
            distance = min(
                (bottom_left_x - other['xmin']) ** 2 + (bottom_left_y - other['ymin']) ** 2,
                (bottom_left_x - other['xmax']) ** 2 + (bottom_left_y - other['ymin']) ** 2,
                (bottom_left_x - other['xmin']) ** 2 + (bottom_left_y - other['ymax']) ** 2,
                (bottom_left_x - other['xmax']) ** 2 + (bottom_left_y - other['ymax']) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_panel = other
        return nearest_panel

    current_bottom = 0  # 現在のページ下端の位置

    while undefined_panels:
        available_panels = []
        # 1. 右上に最も近いコマを探す
        next_panel = find_closest_to_top_right(undefined_panels, current_bottom)
        ordered_panels.append(next_panel)
        undefined_panels.remove(next_panel)

        while not is_leftmost(next_panel, undefined_panels):
            available_panels = find_available_panels(undefined_panels)
            next_panel = find_nearest_panel(next_panel, available_panels)
            ordered_panels.append(next_panel)
            undefined_panels.remove(next_panel)
        # 左端に到達したコマの下端を更新
        current_bottom = next_panel['ymax']

    # 4. 結果を元の形式で返す
    return ordered_panels


if __name__ == '__main__':
    # テストデータ
    data = [
        {'type': 'frame', 'id': '00000009', 'xmin': '899', 'ymin': '585', 'xmax': '1170', 'ymax': '1085'},
        {'type': 'frame', 'id': '0000000c', 'xmin': '2', 'ymin': '0', 'xmax': '826', 'ymax': '513'},
        {'type': 'frame', 'id': '0000000e', 'xmin': '72', 'ymin': '516', 'xmax': '743', 'ymax': '1101'},
        {'type': 'frame', 'id': '00000014', 'xmin': '906', 'ymin': '95', 'xmax': '1575', 'ymax': '576'},
        {'type': 'frame', 'id': '0000001d', 'xmin': '1167', 'ymin': '588', 'xmax': '1580', 'ymax': '1090'}
    ]
    print(panel_order_estimater(data))