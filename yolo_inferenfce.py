from ultralytics import YOLO
model= YOLO("models/best.pt")
results=model.predict("input_videos/video_2024-11-15_15-30-08.mp4",save=True)
print(results[0])
for boxes in results[0].boxes:
    print(boxes)
    
# dikh rha h ki sports ball shi se dey=tect ni ho rhi so we need another model for sport ball and people outside the pitch are also being detected so remove that and refree ka color black h toh usko bhi alag krna h
#toh iske liye ab khud ek train kra h best.pt

# import torch
# print(torch.cuda.is_available())