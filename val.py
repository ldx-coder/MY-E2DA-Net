import matplotlib
matplotlib.use('Agg')


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('E/E2DA-Net-main/runs/demo/weights/best.pt')
    model.val(data='E:/E2DA-Net-main/dataset/dataset.yaml',
              split='val',
              imgsz=640,
              batch=16,
              iou=0.7,
              project='runs/val',
              name='exp',
              )
