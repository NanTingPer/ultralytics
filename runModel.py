from ultralytics import  YOLO
model = YOLO("./11best.pt")
results = model("./茶尺蠖.png")
results[0].show()