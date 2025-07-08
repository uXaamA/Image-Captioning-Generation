# Image Captioning with CNN + LSTM

## 📌 Project Overview

This project implements an **Image Captioning Generation System** using the **Flickr8K** dataset. The model leverages a CNN (ResNet-50) for image encoding and an LSTM network (with attention) for decoding into natural language descriptions.

It is part of a Deep Learning assignment focused on learning and building multi-modal deep learning systems that combine vision and language.

---

## 👤 Author

**Muhammad Usama**
MSc Data Science 
Technical Data Anlayst at Haier Pakistan | Aspiring Data Scientist & Entrepreneur

---

## 📂 Dataset

* **Name**: Flickr8K
* **Total Images**: 8000
* **Captions per Image**: 5
* **Train/Val/Test Split**:

  * Training: 7000 images
  * Validation: 1000 images
  * Testing: 200 images
* **Vocabulary Size**: 8,765 unique words
* **Max Caption Length**: 35 tokens

---

## 🧠 Methodology

### 🔧 Preprocessing

**Image Preprocessing:**

* Resize to 224x224
* Normalize using:

  * `mean = [0.485, 0.456, 0.406]`
  * `std  = [0.229, 0.224, 0.225]`

**Text Preprocessing:**

* Lowercase all captions
* Add `<sos>` and `<eos>` tokens
* Remove special characters
* Tokenize words and build vocabulary

### 🔍 Vocabulary

* Unique words mapped to integer IDs
* Embeddings generated using `nn.Embedding`
* Handled rare words using `<unk>` token

### 🏗️ Model Architecture

**Image Encoder:**

* `ResNet-50` pretrained on ImageNet
* Removed final classification layer

**Text Decoder:**

* 2-layer LSTM with attention
* Concatenated image and word embeddings
* Generated sequence of words (caption)

### ⚙️ Training Setup

| Parameter     | Value        |
| ------------- | ------------ |
| Epochs        | 100          |
| Batch Size    | 128          |
| Learning Rate | 5e-4         |
| Optimizer     | AdamW        |
| Loss          | CrossEntropy |

**Key Techniques:**

* Mixed Precision Training
* Learning Rate Scheduling
* Early Stopping (patience = 5)
* Gradient Scaling

---

## 📈 Results

### 🔬 Performance

| Metric | Training | Validation |
| ------ | -------- | ---------- |
| Loss   | 1.24     | 2.87       |
| BLEU-1 | 0.62     | 0.54       |
| BLEU-4 | 0.31     | 0.23       |

### 🧠 Sample Predictions

**Example 1:**

* Actual: *"A group of people playing volleyball on sandy beach"*
* Predicted: *"People are playing game on beach with net"*

**Example 2:**

* Actual: *"Busy city street with yellow taxis and pedestrians"*
* Predicted: *"Urban road with cars and people walking"*

---

## 🔬 Hyperparameter Analysis

| Learning Rate | Val Loss | Epochs |
| ------------- | -------- | ------ |
| 1e-3          | 3.45     | 28     |
| 5e-4          | 2.87     | 35     |
| 1e-4          | 3.12     | 52     |

---

## 🚧 Challenges & Solutions

* **Overfitting** → Applied dropout (0.5) and early stopping
* **Training time** → Used mixed precision (40% speedup)
* **Rare words** → Added `<unk>` token for words with <5 frequency

---

## 🎯 Conclusion

* Successfully achieved **54% BLEU-1** score on validation
* Model captures core semantics of image content
* Attention helps in focusing on relevant visual regions

### 🚀 Future Improvements

* Use larger datasets like Flickr30K
* Experiment with Transformer-based architectures
* Add Beam Search decoding for better caption fluency

---

## 🧪 Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt
wandb login

# 2. Train the model
python main.py \
  --dataset_path /path/to/dataset \
  --mode train \
  --wandb_entity YOUR_WANDB_USERNAME

# 3. Test the model on custom image
python main.py \
  --mode test \
  --image_path /path/to/image.jpg \
  --wandb_entity YOUR_WANDB_USERNAME
```

---

## 💡 Key Features

* 🔍 Mixed precision training (faster training)
* 📈 Full Weights & Biases logging support
* ⚙️ Multi-GPU support using `DataParallel`
* 📉 LR scheduler, early stopping, ReduceLROnPlateau
* 📊 Graphs: Loss vs Epoch, BLEU vs Epoch
* 🖼️ Caption visualization over test images

---

## 📁 Project Structure

```
project/
├── main.py
├── train.py
├── test.py
├── test_eval.py
├── model.py
├── data_utils.py
├── requirements.txt
├── graphs/
├── weights/            # Checkpoint weights
├── Report.pdf          # Analysis and explanation
└── README.md           # You are here ✨
```

---

Feel free to contribute or extend the project with new ideas! 🚀
