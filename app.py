from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure this model is in your project folder

app = FastAPI()

# Store detection logs
detection_logs = []

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = result.names[int(box.cls[0])]
            
            detections.append({"label": label, "confidence": confidence, "box": [x1, y1, x2, y2]})
            
            # Log detection
            detection_logs.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": label,
                "confidence": confidence
            })
    
    return {"detections": detections, "log": detection_logs[-10:]}  # Return last 10 logs

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
