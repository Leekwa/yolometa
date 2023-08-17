import torch
import ultralytics
from ultralytics import YOLO

print(torch.cuda.is_available())
print('==================================')
print(ultralytics.checks())



if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(data='C:/Users/user/PycharmProjects/YOLO08/Fish-44/data.yaml', imgsz=416, batch=1, epochs=30, device=0)

#수정

#수정쑤정