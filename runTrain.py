'''
下面的代码是训练
'''
from ultralytics import YOLO
from ultralytics.nn.modules import CBAM
import sys
sys.modules["ultralytics.nn.tasks"].CBAM = CBAM

# 加载自定义模型配置
model = YOLO('yolo11.yaml')

results = model.train(
    data='run.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='yolo11_train'
)