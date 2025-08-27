## Disclaimer

> **⚠️ Note: This is an ongoing research work.**
>
> This repository is the official implementation of the paper **"Enhancing Wheat Pest Detection: An Edge-Enhanced Deformable Attention Network Approach"**, which has been submitted to *The Visual Computer*.
### Enhancing Wheat Pest Detection: An Edge-Enhanced Deformable Attention Network Approach
🧠 Core Algorithm: E²DA-Net Framework

This repository contains the official implementation of the E²DA-Net framework, which is designed to address the challenge of detecting morphologically similar wheat pests. The core innovation lies in three collaboratively working modules:

    Edge Enhancement Feature Module (E²FM): Employs the Sobel operator to explicitly enhance the model's ability to capture discriminative pest contours and texture features at the input stage.

    Global-Context Deformable Convolutional Network (GDCN): Integrates a novel Global-Context Coordinate Attention (GCCA) mechanism with deformable convolution. This design enables the receptive field to dynamically adapt to various pest morphological variations. The GCCA module provides precise global guidance for the sampling process of the deformable convolution, ensuring a focus on critical feature regions.

    Superficial Detail Fusion Module (SDFM): Effectively combines high-resolution, detailed information from the backbone network with rich semantic information from the neck network. This fusion prevents the loss of fine-grained details crucial for distinguishing similar species.

These three modules form a comprehensive pipeline that enhances feature discriminability from extraction to utilization, significantly improving detection performance for visually similar pests.



<details open>
<summary>Installation</summary>

1. Create a conda environment (recommended):
```bash
conda create -n E2DANet python=3.8
conda activate E2DANet
```
2. Install PyTorch (Please install the appropriate version for your CUDA driver from pytorch.org. For CUDA 11.3, use):
```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Install dependencies:
```bash
# Core Framework
pip install ultralytics==8.3.112

# Additional dependencies
pip install -r requirements.txt
```


</details>

<details open>
<summary>Usage</summary>


### Dataset Preparation

Place your YOLO-format dataset in the dataset/ directory with the following structure:
```bash
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```
Note: The data.yaml file should contain dataset configuration information including paths and class names.

### Training
```bash
yolo detect train data=dataset/data.yaml model=model/E2DA-Net.yaml epochs=300 imgsz=640
```

### Validation
```bash
yolo detect val data=dataset/data.yaml weights=runs/train/exp/weights/best.pt
```



