import cv2
import numpy as np
import webcolors

def nothing(x):
    pass

def color_to_rgb(color_str):
    """
    把使用者輸入的顏色（可以是英文名稱或 RGB 格式）轉換為 RGB numpy 
    """
    try:
        # 用顏色名稱轉換
        return np.array(webcolors.name_to_rgb(color_str))
    except ValueError:
        # 解析 RGB 格式
        parts = color_str.split(',')
        if len(parts) == 3:
            return np.array([int(p) for p in parts])
        else:
            raise ValueError("plz input a current color, e.g., 'red' or '255,0,0'")

def rgb_to_hsv(rgb):
    """
    用 OpenCV 把 RGB 轉換成 HSV
    OpenCV的 cvtColor 用的是 uint8, 所以要先轉成 uint8 類型
    """
    rgb = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
    return hsv

if __name__ == "__main__":

    # 讀取影像 
    img = cv2.imread("C:\kaiser.png")
    if img is None:
        raise IOError("img loading fail")
    
    # 輸入 欲轉換的色系 與 替換後的目標色系
    source_color_str = input("輸入欲轉換的色系 (ex: red or 255,0,0): ")
    target_color_str = input("輸入替換後的目標色系 (ex: black or 0,0,0): ")
    
    # 轉換顏色字串為 RGB 
    source_rgb = color_to_rgb(source_color_str)
    target_rgb = color_to_rgb(target_color_str)
    
    # 由於 OpenCV 使用 BGR 格式, 所以把 target_rgb 轉為 BGR
    target_bgr = target_rgb[::-1]
    
    # 將欲轉換的顏色轉換到 HSV 空間, 並設定初始閾值範圍
    source_hsv = rgb_to_hsv(source_rgb)
    # 設定初始閾值範圍以 source_hsv 為中心
    # 但是這裡的範圍僅為初始值, 所以把這裡做成可以用滑桿自己調
    init_lower_hue = (int(source_hsv[0].item()) - 10) % 180
    init_upper_hue = (int(source_hsv[0].item()) + 10) % 180

    init_lower = [init_lower_hue, 100, 100]
    init_upper = [init_upper_hue, 255, 255]
    
    # 建立UI、滑桿, 做成可以動態調整 HSV 範圍
    cv2.namedWindow("Adjustments")
    cv2.createTrackbar("Lower Hue", "Adjustments", init_lower[0], 179, nothing)
    cv2.createTrackbar("Lower Sat", "Adjustments", init_lower[1], 255, nothing)
    cv2.createTrackbar("Lower Val", "Adjustments", init_lower[2], 255, nothing)
    cv2.createTrackbar("Upper Hue", "Adjustments", init_upper[0], 179, nothing)
    cv2.createTrackbar("Upper Sat", "Adjustments", init_upper[1], 255, nothing)
    cv2.createTrackbar("Upper Val", "Adjustments", init_upper[2], 255, nothing)
    
    while True:
        # 從滑桿取得 HSV 閾值
        lh = cv2.getTrackbarPos("Lower Hue", "Adjustments")
        ls = cv2.getTrackbarPos("Lower Sat", "Adjustments")
        lv = cv2.getTrackbarPos("Lower Val", "Adjustments")
        uh = cv2.getTrackbarPos("Upper Hue", "Adjustments")
        us = cv2.getTrackbarPos("Upper Sat", "Adjustments")
        uv = cv2.getTrackbarPos("Upper Val", "Adjustments")
        
        lower_bound = np.array([lh, ls, lv])
        upper_bound = np.array([uh, us, uv])
        
        # 把影像轉換到 HSV 空間
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 判斷是否需要處理跨越色相 0度 的情況（例如紅色）
        if uh >= lh:
            # 若 上界 >= 下界, 直接使用單一範圍
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            # 跨越0度: 例如 lower hue = 170, upper hue = 10
            mask1 = cv2.inRange(hsv, np.array([0, ls, lv]), np.array([uh, us, uv]))
            mask2 = cv2.inRange(hsv, np.array([lh, ls, lv]), np.array([179, us, uv]))
            mask = cv2.bitwise_or(mask1, mask2)
        
        # 把遮罩選中的區域替換為目標顏色
        result = img.copy()
        result[mask != 0] = target_bgr
        
        cv2.imshow("Result", result)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出網路紅人
            break

    cv2.destroyAllWindows()
