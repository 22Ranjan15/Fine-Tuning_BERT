# BART-based SMS Spam Classifier

[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Transformers-%23FFD21F.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)


A fine-tuned BERT model for classifying SMS messages as spam or ham. This project demonstrates the process of using a pre-trained BERT model for text classification.


## Overview
This repository contains a BERT-based text classification model fine-tuned for spam detection.  The model is trained on the spamdata_v2.csv dataset and demonstrates the effectiveness of BERT for distinguishing between spam and legitimate (ham) SMS messages.

## Dataset
The model is trained on the [SMS Spam Collection Dataset](https://github.com/22Ranjan15/Fine-Tuning_BERT/blob/main/spamdata_v2.csv) which contains:
- 5,572 SMS messages
- Class balance: spam (13.4%) / ham (86.6%)
- Raw text data with original category labels

## Features

This BERT-based spam classifier offers the following capabilities:

* **Accurate Spam Detection:**

    * Leverages fine-tuned BERT (`bert-base-uncased`) to effectively classify SMS messages.

    * Provides strong performance in distinguishing between spam (1) and ham (0) messages.
        * Overall Accuracy: 97%
        * Precision (Spam): 99%
        * Recall (Spam): 97%
        * F1-Score (Spam): 98%

* **BERT-Powered NLP:**
    * Employs the `bert-base-uncased` model, capturing contextual relationships within text for improved classification.

    * Utilizes the `transformers` library for efficient tokenization and model handling.

* **PyTorch Implementation:**
    * Built using PyTorch for flexible model training and execution.

    * Includes DataLoaders for optimized batching and data loading during training.

* **Detailed Workflow:**
    * The provided notebook (`Fine_Tuning_BERT_for_Spam_Classification.ipynb`) demonstrates the complete process:

        * Data loading and preprocessing
        * Model fine-tuning
        * Evaluation with classification reports and confusion matrix.

## Requirements

- Python 3.7+ (Based on common usage with the specified libraries)
- Dependencies (install using pip):
  ```bash
  transformers>=4.0.0
  torch>=1.6.0
  pandas>=1.1.0
  scikit-learn>=0.23.0
  numpy>=1.19.0
  ```

# Quick Start
```python
from transformers import pipeline

# Load the trained model
classifier = pipeline("text-classification", model="spam_classifier")

# Example prediction
sample_message = "WINNER! Claim your free iPhone now! Text YES to 12345."
result = classifier(sample_message)

print(f"Message: {sample_message}")
print(f"Prediction: {result[0]['label']} (confidence: {result[0]['score']:.2f})")
```

# Clone this repository:

```bash
git clone https://github.com/22Ranjan15/Fine-Tuning_BERT.git
```

# Training
To train the model from scratch:
```python
Fine-Tune BERT_Spam Classification.ipynb
```

## Model Performance

| Metric | Ham Class | Spam Class | Overall |
|--------|-----------|------------|---------|
| Accuracy | - | - | 97% |
| Precision | 83% | 99% | - |
| Recall | 95% | 97% | - |
| F1-Score | 89% | 98% | - |


## How It Works

The model uses a pre-trained BART model as the base
It's fine-tuned on the SMS Spam Collection dataset
BART's sequence classification capabilities are leveraged for binary classification
The model learns to identify patterns and language typically associated with spam messages

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

# Acknowledgments
- Dataset provided by UCI Machine Learning Repository

- Hugging Face for the Transformers library

- Facebook AI Research for the original BART implementation
