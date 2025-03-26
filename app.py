import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

import numpy as np
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# 加载 YOLOv8 模型（需替换为你的模型路径）
model = YOLO('ckpt/YOLOv10x_gestures.pt', verbose=False)  # 确保模型已训练过“手部合十”类别

# 状态变量
is_praying = False
start_time = 0


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('video_frame')
def process_frame(data):
    import base64
    base64_str = data['frame']

    # 解码为字节数据
    img_bytes = base64.b64decode(base64_str)  # bytes类型
    """接收前端发送的视频帧，进行推理并返回结果"""
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 进行推理
    results = model(frame)[0]
    target_cls_name = "holy"
    target_cls_id = list(model.names.keys())[list(model.names.values()).index(target_cls_name)]
    filtered_boxes = [box for box in results.boxes if (int(box.cls) == target_cls_id) and (box.conf >= 0.70)]
    detected = False if len(filtered_boxes) == 0 else True

    # 返回检测结果（是否合十）
    emit('detection_result', {'detected': detected})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
