# export_onnx.py
'''
如何使用CBAM
1. 在 conv.py的CBAM中编写CBAM模块的代码
2. 在ultralytics的__init__中导入 CBAM
3. 在ultralytics.cfg.models.11.yolo11中插入CBAM
4. 使用yolo11.yaml加载模型配置
'''

'''
下面是导入
'''
from ultralytics.nn.modules import CBAM

print("Hello")
from ultralytics import YOLO

model = YOLO("./best_noCBAM.pt")
model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False,
    imgsz=640,
    device="cpu",  # 明确指定 CPU
)