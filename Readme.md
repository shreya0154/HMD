# Hateful Meme Detection Project

## Project Overview
This project aims to detect hateful memes by analyzing both visual and textual data. It utilizes state-of-the-art machine learning models such as Qwen-2-VL-7B Instruct, YOLOv8, and DeepFace for feature extraction and classification. The ensemble of transformers ensures robust accuracy in detecting hateful content.

---

## Dataset
- **Name**: Hateful Memes Dataset (Facebook AI)
- **Content**: Contains over 10,000 labeled examples combining images and text.
- **Source**: Getty Images (licensed for research purposes).
- **Purpose**: Designed for advancing multimodal hate speech detection.

---

## Libraries Required
Install the following Python libraries before running the code:
- **Deep Learning**:
  - `torch` (PyTorch)
  - `tensorflow` (TensorFlow)
  - `transformers` (Hugging Face)
- **Computer Vision**:
  - `opencv-python`
  - `ultralytics` (for YOLOv8)
  - `deepface`
- **Data Processing and Visualization**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`

---

## Environment Details
- **Operating System**: Linux/Windows/macOS
- **Hardware Requirements**:
  - **Minimum**: 8 GB RAM, NVIDIA GPU with 4 GB VRAM
  - **Recommended**: 16 GB RAM, NVIDIA GPU with 8+ GB VRAM
- **Python Version**: 3.9 or higher
- **GPU Support**: CUDA-enabled GPU is recommended for faster inference.

---

## Instructions for Running the Code (UI)

1. **Install required libraries**
  !pip install ipywidgets
  !pip install deepface
  !pip install ultralytics
  !pip install git+https://github.com/huggingface/transformers

2. **Clone the Repository**
  git clone https://github.com/blackhat-coder21/Mini_Project_Sem5
  cd Mini_Project_Sem5
  streamlit run app.py
