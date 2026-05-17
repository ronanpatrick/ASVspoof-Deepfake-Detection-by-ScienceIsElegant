# Comparative Analysis and Architectural Ablation of 1D and 2D CNNs for Audio Deepfake Detection

**Created by: Group Science is Elegant**

## Overview
This repository contains a complete deep learning codebase for detecting AI-generated synthetic speech (deepfakes). The project evaluates and compares the performance of two distinct neural network architectures on the **ASVspoof 2019 Logical Access (LA)** dataset, exploring how different data representations and architectural constraints impact a model's ability to generalize deepfake artifacts.

## Architectures Evaluated
1. **2D ResNet-18:** Processes audio converted into 2D Mel-Spectrograms. An ablation study was conducted by introducing a 50% Dropout layer to observe unregularized generalization vs. memorization.
2. **1D RawNet2:** Processes raw, uncompressed 1D audio waveforms. An ablation study was conducted by expanding the GRU dimensionality from 64 to 128 to test temporal memory capacity.

## Final Results
To evaluate the raw computational capacity of the architectures, all models were trained for a fixed 100 epochs without dynamic early stopping. 

| Model Architecture | Input Feature | Evaluation EER (%) |
| :--- | :--- | :--- |
| **2D ResNet-18 (Baseline)** | Mel-Spectrogram | **11.2%** |
| 2D ResNet-18 (Tweaked - 50% Dropout) | Mel-Spectrogram | 11.6% |
| **1D RawNet2 (Tweaked - 128 GRU)** | Raw Waveform | **23.2%** |
| 1D RawNet2 (Baseline) | Raw Waveform | 26.8% |

## Preliminary Observations
* **2D Architecture Performance:** In the fixed 100-epoch training setup, the **2D ResNet-18 (Baseline)** achieved the lowest overall Equal Error Rate at 11.2%. Introducing a 50% Dropout layer (Tweaked) resulted in a closely comparable EER of 11.6%.
* **1D Architecture Performance:** For the raw waveform models, expanding the GRU capacity from 64 to 128 dimensions resulted in a performance improvement, dropping the EER from 26.8% (Baseline) to 23.2% (Tweaked).

## Usage & Dataset Disclaimer
Due to GitHub's strict file size limits, the ASVspoof 2019 LA dataset (.flac files) and the generated .npy feature arrays are not included in this repository. To run the provided Jupyter notebook locally:
1. Download the LA dataset from the official [ASVspoof 2019 website](https://www.asvspoof.org/).
2. Update the `BASE_DIR` paths in Phase 1 of the notebook to point to your local directories.

## Technologies Used
* PyTorch
* Librosa (Audio Processing)
* Pandas & NumPy
