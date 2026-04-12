from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640)
print("변환 완료: yolov8n.onnx")