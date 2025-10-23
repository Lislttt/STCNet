# STCNet: A hybrid Transformer and CNN network for surface defect detection of Steel Continuous Casting Billets
Welcome to the official repository for STCNet. This repository includes experiments and model of the papper [STCNet: A hybrid Transformer and CNN network for surface defect detection of Steel Continuous Casting Billets].

Authors: Shanlin Li, Zhehan Chen, Xiaodong Zhang, Tao Yang and Dianning Gong

![image](https://github.com/Lislttt/STCNet/blob/main/overview.png)

## Description

STCNet is a lightweight detection model for steel continuous casting billet surface defects. It combines a star-operation backbone, convolutional additive self-attention, and adaptive cross-scale fusion to enhance feature learning while maintaining efficiency. Evaluated on the SCCB-SDD and GC10-DET datasets, STCNet achieves a strong balance between accuracy and computational cost, outperforming mainstream methods with only 7.1M parameters and 15.6 GFLOPs. This repository provides [model](https://github.com/Lislttt/STCNet/blob/main/Model.py) and [checkpoints](https://drive.google.com/drive/folders/14mQgwJL4pAPDoDQMq7B_8YtEvE86XrMy?usp=sharing) to reproduce proposed method.


## Datesets

The proposed SCCB-SDD dataset comprises 7,398 images with resolutions around 320×320, covering seven representative types of surface defects in steel continuous casting billets, along with background samples. Specifically, the dataset includes background (1,112 images), roll mark (645), transverse crack (1,112), scratches (773), peeling off (1,112), iron oxide scale (1,112), slag skin (420), and longitudinal crack (1,112).

On this dataset, the proposed STCNet achieves an outstanding mAP@50 of 92.1%and mAP@75 of 92.1%, surpassing a broad range of state-of-the-art CNN- and Transformer-based object detection models.
Among CNN-based methods, YOLOv11 and YOLOv8 achieve 90.4% and 88.8% mAP@50, respectively, while classical two-stage detectors such as Faster R-CNN (80.5%) and SSD perform considerably worse.
Compared with Transformer-based detectors, STCNet also outperforms RT-DETR (90.1%) and Swin Transformer (90.3%), and achieves accuracy comparable to DN-Deformable-DETR (92.0%), yet with 7× fewer parameters (7.1M vs. 47.2M) and only 15.9 GFLOPs of computation. Moreover, STCNet reaches an inference speed of 142.9 FPS, approximately 6× faster than most DETR variants, demonstrating its superior balance between precision and efficiency.

These results clearly demonstrate that STCNet not only attains state-of-the-art detection accuracy, but also delivers remarkable computational efficiency. The hybrid Transformer–CNN architecture enables global contextual modeling through the Transformer encoder while preserving fine-grained local texture representation via CNN feature extraction, resulting in robust and real-time detection of small-scale and low-contrast defects in industrial steel billet inspection scenarios.

## Reference

@article{10.1088/1361-6501/ae140a,<br>
	author={Li, Shanlin and Chen, Zhehan and Zhang, Xiaodong and Yang, Tao and Gong, Dianning},<br>
	title={STCNet: A hybrid Transformer and CNN network for surface defect detection of Steel Continuous Casting Billets},<br>
	journal={Measurement Science and Technology},<br>
	url={http://iopscience.iop.org/article/10.1088/1361-6501/ae140a},<br>
	year={2025},<br>
	abstract={In steel manufacturing, steel continuous casting billets are essential intermediate products, making surface defect detection a critical task. To address the challenge of limited industrial datasets that hinder the accuracy of deep learning methods, this paper introduces the Steel Continuous Casting Billets Surface Defect Detection (SCCB-SDD) dataset, constructed from real production data. We also propose STCNet, a novel detection model that combines Transformer and Convolutional Neural Network (CNN) architectures. Designed for industrial applications, STCNet employs a star-operation backbone for feature extraction. It further incorporates an intra-scale interaction encoder with convolutional additive self-attention to strengthen same-scale feature learning. An adaptive cross-scale fusion encoder integrates multi-scale information effectively. In our experiments, raw images from the SCCB-SDD dataset were manually annotated, with defect regions cropped into compact, information-dense samples to reduce background noise and accelerate model convergence. Results show that STCNet outperforms mainstream competing methods, achieving a mean Average Precision (mAP@50) of 92.1% with only 7.1 million parameters and 15.9 GFLOPs (Giga Floating Point Operations per Second). Furthermore, to validate the generalization capability of the model, we conducted evaluations on the GC10-DET dataset, where STCNet achieved 66.4% mAP@50 and 30.1% mAP@75, outperforming mainstream competing methods. These results demonstrate that STCNet achieves a strong balance between accuracy and efficiency, emphasizing its practical value for industrial billet defect detection.}
}
