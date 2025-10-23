# STCNet: A hybrid Transformer and CNN network for surface defect detection of Steel Continuous Casting Billets
Welcome to the official repository for STCNet. This repository includes experiments and model of the paper [STCNet: A hybrid Transformer and CNN network for surface defect detection of Steel Continuous Casting Billets].

Authors: Shanlin Li, Zhehan Chen, Xiaodong Zhang, Tao Yang and Dianning Gong

![image](https://github.com/Lislttt/STCNet/blob/main/overview.png)

## Description

STCNet is a lightweight detection model for steel continuous casting billet surface defects. It combines a star-operation backbone, convolutional additive self-attention, and adaptive cross-scale fusion to enhance feature learning while maintaining efficiency. Evaluated on the SCCB-SDD and GC10-DET datasets, STCNet achieves a strong balance between accuracy and computational cost, outperforming mainstream methods with only 7.1M parameters and 15.6 GFLOPs. This repository provides [model](https://github.com/Lislttt/STCNet/blob/main/Model.py) and [checkpoints](https://drive.google.com/drive/folders/14mQgwJL4pAPDoDQMq7B_8YtEvE86XrMy?usp=sharing) to reproduce proposed method.


## Datesets

The proposed [SCCB-SDD](https://drive.google.com/file/d/17nEcTiUuU_aPd1jO-7V15pqspDomW75l/view?usp=drive_link) dataset comprises 7,398 images with resolutions ranging from 128×128 to 640×640, covering seven representative types of surface defects in steel continuous casting billets, along with background samples. Specifically, the dataset includes background (1,112 images), roll mark (645), transverse crack (1,112), scratches (773), peeling off (1,112), iron oxide scale (1,112), slag skin (420), and longitudinal crack (1,112).

On this dataset, the proposed STCNet achieves an outstanding mAP@50 of 92.1%and mAP@75 of 92.1%, surpassing a broad range of state-of-the-art CNN- and Transformer-based object detection models.
Among CNN-based methods, YOLOv11 and YOLOv8 achieve 90.4% and 88.8% mAP@50, respectively, while classical two-stage detectors such as Faster R-CNN (80.5%) and SSD perform considerably worse.
Compared with Transformer-based detectors, STCNet also outperforms RT-DETR (90.1%) and Swin Transformer (90.3%), and achieves accuracy comparable to DN-Deformable-DETR (92.0%), yet with 7× fewer parameters (7.1M vs. 47.2M) and only 15.9 GFLOPs of computation. Moreover, STCNet reaches an inference speed of 142.9 FPS, approximately 6× faster than most DETR variants, demonstrating its superior balance between precision and efficiency.

Overall, these results indicate that STCNet achieves competitive detection accuracy while maintaining high computational efficiency. The hybrid Transformer–CNN architecture effectively combines global contextual modeling through the Transformer encoder with fine-grained local texture representation via CNN feature extraction, enabling robust and real-time detection of small-scale and low-contrast defects in industrial steel billet inspection scenarios.

![image](https://github.com/Lislttt/STCNet/blob/main/overview.png)

## Reference

@article{10.1088/1361-6501/ae140a,<br>
	author={Li, Shanlin and Chen, Zhehan and Zhang, Xiaodong and Yang, Tao and Gong, Dianning},<br>
	title={STCNet: A hybrid Transformer and CNN network for surface defect detection of Steel Continuous Casting Billets},<br>
	journal={Measurement Science and Technology},<br>
	url={http://iopscience.iop.org/article/10.1088/1361-6501/ae140a},<br>
	year={2025}
}
