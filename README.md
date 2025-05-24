# 🧠 Evaluation of generalization of Convolutional Neural Networks trained with proxy tasks

📌 This repository contains the code, pretrained models, and documentation related to my undergraduate thesis on **Self-Supervised Learning** using proxy tasks. The goal of this project was to evaluate how effectively various proxy tasks can train convolutional neural networks without labeled data, and how well the learned representations transfer to downstream tasks like image retrieval and classification

## 🧾 Abstract
This repo
In recent years, Deep Learning (DL) and its applications in Computer Vision (CV) have seen rapid advancements. Despite this progress, many datasets remain underutilized due to the lack of annotations (unlabeled data). Labeling these datasets is a time-intensive process that often requires specialized human intervention, which makes them difficult to leverage effectively. Self-Supervised Learning (SSL) provides a promising solution to this problem by utilizing proxy/pretext tasks that train models without requiring labeled data. In these tasks, the model generates its own labels by manipulating the data, enabling it to learn meaningful features like object orientation, color, and image coherence, which can then be applied to tasks such as Image Classification or Object Detection.

This thesis aims to compare three different proxy tasks: Image Rotation Prediction, Image Colorization, and Image Inpainting. The objective is to establish a fair comparison environment to evaluate the effectiveness of these tasks in the downstream task of image classification, which is inherently different as it requires labeled data for training. To ensure a fair comparison, the same architecture, hyperparameters, and settings were consistently applied across all proxy tasks and the downstream task.
For the implementation of the three proxy tasks, the ResNet-50 model was used as the base architecture, one of the most efficient and popular models in Convolutional Neural Networks (CNNs). The model was adapted to suit the specific requirements of each task. The key principle of the ResNet-50’s architecture is the use of "skip connections," which allow the transmission of information from earlier layers to subsequent layers, helping to prevent the vanishing gradient phenomenon. The pretrained weights obtained from these proxy tasks were transferred via Transfer Learning and were then evaluated on the downstream task of image classification.

For the training of the proxy tasks, the CIFAR-100 dataset was used as an unlabeled dataset, meaning only the images without their labels were utilized. For the downstream task of image classification, the CIFAR-10 and CIFAR-100 datasets were used. In the downstream task, fine-tuning was performed in two different ways: one involved fine-tuning only the last fully connected layer of the model to compare the information learned by the pretrained weights against random weights. In the second case, fine-tuning was done on all layers of the model to assess whether fully adapting the pretrained weights yields better performance than random weights.

In the experimental part of the thesis, the experimental results and metrics showed that Self-Supervised Learning through proxy tasks offers significantly better performance compared to training with randomly initialized weights. Additionally, the model was able to achieve a high level of performance, close to its peak performance, which would typically be reached after many more epochs, making the process faster and more efficient. The most effective proxy task was Image Inpainting, followed closely by Image Colorization. Image Rotation Prediction had the lowest performance, but still contributed to improving the model's performance compared to random weights.


The thesis includes both:

* 📖 A **theoretical part**: explanation of the concepts behind cnns, deep learning, self-supervised learning and proxy tasks.
* 🧪 An **experimental part**: implementation, training, evaluation results, commentary of results and conclusions.

## 📁 Repository Structure

```text
📦 root/
├── README.md                           # Main project README
├── thesis/                             # Thesis documents
│   ├── Thesis_Experimental_EN.pdf      # Thesis experimental part in English
│   ├── Thesis_Experimental_GR.pdf      # Thesis experimental part in Greek
│   ├── Thesis_Full_GR.pdf              # Full thesis (theory + experiments) in Greek
│   └── Thesis_Presentation_GR.pdf      # Presentation slides in Greek
│
├── src/                                # Source code for experiments
│   ├── proxy_tasks/                    # Self-supervised proxy tasks
│   │   ├── rotation/                   # Image Rotation Prediction implementation
│   │   ├── colorization/               # Image Colorization implementation
│   │   └── inpainting/                 # Image Inpainting implementation
│   │
│   └── downstream_tasks/               # Downstream task evaluations
│       ├── cifar_10_classification_final_layer/       # Fine-tuning only the last layer on CIFAR-10
│       ├── cifar_10_classification_whole_model/       # Fine-tuning the entire model on CIFAR-10
│       ├── cifar_100_classification_final_layer/      # Fine-tuning only the last layer on CIFAR-100
│       └── cifar_100_classification_whole_model/      # Fine-tuning the entire model on CIFAR-100
│
└── weights/                            # Pretrained weights from proxy tasks
    ├── cifar100-rotation-bestmodel.pth            # Best model from the rotation task on CIFAR-10
    ├── cifar100_colorization_bestmodel.pth        # Best model from the colorization task on CIFAR-100
    └── cifar100_inpainting_bestmodel.pth          # Best model from the inpainting task on CIFAR-10
```

## 🔁 Usage

You can run each proxy task from scratch using the code in `src/proxy_tasks/`, or skip training and use the pretrained models provided in the `weights/` directory to directly run the downstream evaluations in `src/downstream_tasks/`.

Each downstream folder corresponds to fine-tuning on either CIFAR-10 or CIFAR-100, using either just the final layer or the whole model.

