# Animal Image Classification


## Project Overview

This repository contains an image classification system that identifies animals from images. Leveraging deep learning and transfer learning, the model classifies images into 15 animal categories with high accuracy.

<br/>

## 🚀 Objective

* Develop a CNN-based classifier to recognize animals in images.
* Explore data preprocessing and augmentation techniques.
* Utilize transfer learning (MobileNetV2) for faster convergence and improved performance.

<br/>

## 📂 Dataset

* **Location:** `data/` directory
* **Structure:** 15 subfolders, each named after one of the classes:

  * Bear
  * Bird
  * Cat
  * Cow
  * Deer
  * Dog
  * Dolphin
  * Elephant
  * Giraffe
  * Horse
  * Kangaroo
  * Lion
  * Panda
  * Tiger
  * Zebra
* **Image Size:** 224 × 224 × 3 pixels

<br/>


## 🔧 Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/amareshmaity/Deep-Learning-Projects.git
   cd Deep-Learning-Projects/Animal-Image-Classification
   ```

2. **Create Virtual Environment (optional)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

<br/>

## 🏃‍♂️ Usage

### 1. Training the Model

```bash
python src/train.py \
  --data_dir data \
  --epochs 60 \
  --batch_size 32 \
  --save_dir saved_models
```

* **--data\_dir:** Path to `data/` directory
* **--epochs:** Number of training epochs
* **--batch\_size:** Batch size for training
* **--save\_dir:** Directory to save the best model

<br/>

### 2. Evaluating the Model

```bash
python src/evaluate.py \
  --model_path saved_models/best_model.h5 \
  --data_dir data
```

### 3. Predicting a Single Image

```python
from src.model_builder import load_trained_model, predict_image

model = load_trained_model('saved_models/best_model.h5')
result = predict_image(model, 'path/to/image.jpg')
print(f"Predicted class: {result}")
```

---

## 📈 Results

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | XX.XX% |

* **Top Classes:** Dolphin, Giraffe
* **Challenging Classes:** Bird, Zebra

---

## 🔍 Methodology

1. **Preprocessing & Augmentation**

   * Rescale pixel values to \[0,1].
   * Apply random rotations, zooms, flips.
   * Split into 70% train, 15% val, 15% test.

2. **Model Architecture**

   * Base: MobileNetV2 pretrained on ImageNet (include\_top=False).
   * Add: GlobalAveragePooling2D → Dropout(0.4) → Dense(15, softmax).

3. **Training Strategy**

   * Phase 1: Train top layers (base frozen).
   * Phase 2: Unfreeze last blocks, fine-tune with lower LR.
   * Optimizer: Adam; Loss: CategoricalCrossentropy; EarlyStopping & ModelCheckpoint.

---

## 📚 References

* TensorFlow Transfer Learning Tutorial: [https://www.tensorflow.org/tutorials/images/transfer\_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
* Keras ImageDataGenerator: [https://keras.io/api/preprocessing/image/](https://keras.io/api/preprocessing/image/)
* MobileNetV2 Paper: Howard et al., CVPR 2018

---

### 👤 Created by -

**Amaresh Maity**
* GitHub: [amareshmaity](https://github.com/amareshmaity)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
