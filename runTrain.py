'''
下面的代码是训练
'''
from ultralytics import YOLO
# 加载自定义模型配置
model = YOLO('yolo11.yaml')

results = model.train(
    data='run.yaml',
    epochs=10,
    imgsz=640,
    batch=8,
    name='yolo11_train'
)