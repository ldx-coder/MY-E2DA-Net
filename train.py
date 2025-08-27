import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings



if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/E2DA-Net.yaml')
    model.train(data='E:/E2DA-Net-main/dataset/dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=0,
                workers=0,
                optimizer='SGD',
                project='runs/',
                name='demo',
                )
