from ultralytics import YOLO
import cv2
import time

model = YOLO("./ckpt/YOLOv10x_gestures.pt")
cap = cv2.VideoCapture(0)  # 摄像头输入，或替换为视频路径

# 设置帧率（FPS）
cap.set(cv2.CAP_PROP_FPS, 30)  # 目标帧率为30

# 设置分辨率（清晰度）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # 宽度为480像素
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # 高度为360像素

# 初始化计时器
timer_started = False
start_time = None
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % 10 == 0:
        results = model(frame)

        # 提取检测框信息
        boxes = results[0].boxes

        # 过滤条件：仅保留"holy"（假设"holy"类别ID为某个值）
        target_cls_name = "holy"
        target_cls_id = list(model.names.keys())[list(model.names.values()).index(target_cls_name)]
        filtered_boxes = [box for box in boxes if (int(box.cls) == target_cls_id) and (box.conf >= 0.70)]

        # 找到置信度最高的目标
        if len(filtered_boxes) > 0:
            # 按置信度排序，取第一个（最高）
            highest_conf_box = sorted(filtered_boxes, key=lambda x: x.conf, reverse=True)[0]

            # 提取坐标和置信度
            x1, y1, x2, y2 = map(int, highest_conf_box.xyxy[0].tolist())
            conf = highest_conf_box.conf.item()

            # 绘制框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"holy {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 启动计时器
            if not timer_started:
                start_time = time.time()
                timer_started = True

            # 检查是否已经过了30秒
            if timer_started and (time.time() - start_time) >= 15:
                print("finish")
                timer_started = False
                break
        else:
            # 如果中断了，输出"break"
            if timer_started:
                print("break")
                timer_started = False

    cv2.imshow("YOLOv10 Real-time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
