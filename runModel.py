from ultralytics import  YOLO
model = YOLO("./11best.pt")
results = model("./chongzi.png")
results[0].show()