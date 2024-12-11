import cv2
import numpy as np

# 加载视频
cap = cv2.VideoCapture('video.mp4')

# 背景减法器
fgbg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=50, detectShadows=True)

# 形态学操作的核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# 宽高阈值
min_width = 50
min_height = 50

# 定义一个虚拟线，用于统计车辆数量
line_position = 180  # 线的纵坐标位置
offset = 8  # 容许的偏移量，用于判断车辆是否通过线

# 车辆计数变量
vehicle_count = 0

# 存储车辆的中心点和帧号
detected_centers = []  

# 冷却时间，防止重复计数（单位：帧）
cooldown_time = 10

# 当前帧号
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 增加帧号计数
    frame_number += 1

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用背景减法
    fgmask = fgbg.apply(frame)
    
    # 阈值化以去除噪声
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    # 形态学操作
    # 1. 去除噪声（开运算）
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 2. 填补空洞（闭运算）
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel1)
    # 3. 膨胀操作以增强目标区域
    fgmask = cv2.dilate(fgmask, kernel1, iterations=3)

    # 查找轮廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓并绘制
    for contour in contours:
        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 根据宽高阈值过滤
        if w >= min_width and h >= min_height:
            # 计算车辆的中心点
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # 绘制外接矩形和中心点
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # 检测车辆是否通过虚拟线
            if (line_position - offset) < center_y < (line_position + offset):
                # 检查是否是新车辆（冷却时间内不重复计数）
                is_new_vehicle = True
                for center in detected_centers:
                    old_x, old_y, old_frame = center
                    if abs(center_x - old_x) < 20 and abs(center_y - old_y) < 20 and (frame_number - old_frame) < cooldown_time:
                        is_new_vehicle = False
                        break

                # 如果是新车辆，计数并记录
                if is_new_vehicle:
                    vehicle_count += 1
                    detected_centers.append((center_x, center_y, frame_number))

    # 清理过期数据（超过冷却时间的记录）
    detected_centers = [center for center in detected_centers if (frame_number - center[2]) < cooldown_time]

    # 绘制虚拟线
    cv2.line(frame, (80, line_position), (800, line_position), (0, 0, 255), 3)

    # 显示车辆计数
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Original Frame', frame)
#    cv2.imshow('Foreground Mask', fgmask)

    # 按 'q' 退出
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
