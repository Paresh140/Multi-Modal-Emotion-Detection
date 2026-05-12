# Multimodal Emotion Recognition: Fusing Speech and Text

This repository implements a deep learning framework that fuses acoustic and semantic features for robust emotion recognition. By combining **Wav2Vec 2.0** for audio and **RoBERTa** for text, the model captures both prosodic and linguistic cues to improve classification accuracy.

## 🚀 Overview
Emotion recognition is a challenging task because neither words nor tone alone provide the full context. This project implements a **late-fusion multimodal model** evaluated on the **CREMA-D** dataset. The system achieves significant performance gains by leveraging the complementary nature of audio and text signals.

## 🏗️ Model Architecture
The architecture consists of two specialized transformer-based encoders fused at the final stage:

* **Audio Encoder:** Based on `wav2vec2-base`. It processes raw 1-D waveforms. The convolutional feature extractor and the bottom 8 transformer layers are frozen to preserve pre-trained representations, while the top 4 layers are fine-tuned. Mean pooling produces a 768-dimensional utterance embedding.
* **Text Encoder:** Based on `roberta-base`. It extracts semantic features from ground-truth transcriptions using the `[CLS]` token as the global sequence representation. This produces a second 768-dimensional embedding.
* **Late Fusion:** Both embeddings are concatenated into a single 1,536-dimensional vector.
* **Classification Head:** A two-layer MLP consisting of a dropout layer (rate 0.3), a linear projection (1,536 to 256), ReLU activation, a second dropout layer, and a final linear layer mapping to the 6 emotion classes.



## 📊 Dataset & Evaluation
* **Dataset:** CREMA-D (7,442 utterances from 91 professional actors).
* **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad.
* **Protocol:** Strict speaker-independent 5-fold cross-validation.
* **Data Split:** Approximately 71% training (5,316 samples), 14% validation (1,060 samples), and 14% testing (1,066 samples).

## 📈 Results
The multimodal approach consistently outperforms unimodal baselines:

| Model | Val. Accuracy | Test Accuracy | Macro-F1 |
| :--- | :--- | :--- | :--- |
| Audio-only (Wav2Vec 2.0) | 62.3% | 60.1% | 0.590 |
| Text-only (RoBERTa) | 68.1% | 65.9% | 0.650 |
| **Multimodal Fusion (Best Fold)** | **75.09%** | **69.79%** | **0.696** |

### Key Findings
* **Performance Gain:** The fusion model exceeds the text-only baseline by **+4.1%** and the audio-only baseline by **+9.7%** in test accuracy.
* **Class Specifics:** "Angry" (84.5% recall) and "Neutral" (87.1% recall) are the most reliably detected classes.
* **Error Patterns:** "Disgust" (54.4% recall) is frequently confused with "Sad" due to shared low-energy acoustic profiles.

## 🛠️ Training Setup
* **Hardware:** NVIDIA H100 80 GB GPU.
* **Optimizer:** AdamW with a decoupled weight decay of $1 \times 10^{-4}$.
* **Learning Rates:** $2 \times 10^{-5}$ for the audio and text encoders; $1 \times 10^{-4}$ for the classification head.
* **Schedule:** Linear warmup (first 5% of steps) followed by a cosine decay schedule.
* **Batch Configuration:** Physical batch size of 8 with gradient accumulation over 8 steps, resulting in an effective batch size of 64.
