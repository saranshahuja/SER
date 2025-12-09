# Speech Emotion Recognition Experiments

This repository collects experiments for speech emotion recognition using a mix of convolutional, recurrent, attention, and transformer-based models. The code is organized primarily as Jupyter notebooks so you can reproduce and extend the workflows that produced the accompanying research paper published in IEEE Access ([doi:10.1109/ACCESS.2023.3303179](https://ieeexplore.ieee.org/document/10195078)).

## Repository contents
- **`TL_VGG_RAVDESS.ipynb`** & **`TL_VGGish_speech.ipynb`**: Transfer learning experiments that adapt VGG and VGGish feature extractors to emotion classification.
- **`parallel_cnn_transformer.ipynb`** & **`parallel_cnn_attention_lstm.ipynb`**: Hybrid architectures that blend convolutional front ends with transformer or attention-augmented LSTMs.
- **`stacked_cnn_lstm.ipynb`** & **`stacked_cnn_attention_lstm.ipynb`**: Deep CNN stacks feeding into recurrent layers for temporal context modeling.
- **`cnn_LTSM.ipynb`**: Baseline convolutional + LSTM model (typo preserved from the original experiment naming).
- **`dataAugmentationforSER.ipynb`**: Exploration of audio augmentation strategies to improve robustness across datasets.
- **`SAVEE Test.ipynb`** and other dataset-specific notebooks: Quick evaluations on individual corpora such as SAVEE and RAVDESS.
- **`SpeechEmotionRecognitionCNN.py`**: Minimal Python script showing dataset path configuration and emotion label mapping for experimentation outside notebooks.
- **`training_output.txt`**: Example training log captured from one of the notebook runs.

## Getting started
1. **Create an environment** with Python 3.9+ and install dependencies used across the notebooks:
   ```bash
   pip install torch torchvision torchaudio librosa pandas numpy matplotlib ipykernel
   ```
2. **Launch Jupyter** in the repository root and open the notebook that matches the architecture you want to explore:
   ```bash
   jupyter notebook
   ```
3. **Configure data paths** inside each notebook or in `SpeechEmotionRecognitionCNN.py` to point to your local dataset folders before running cells.

## Reproducing results
- Run the notebooks sequentially, ensuring the dataset locations and sampling rates align with your local copies. Many cells will visualize spectrograms and play back audio to aid qualitative assessment.
- Track training output (e.g., accuracy and loss) using the printed logs or by extending the notebooks to log to TensorBoard.

## Citing this work
If you build on these experiments, please cite the associated paper:

> S. [Author Names], "Temporal Parallel CNN Transformer Model with Integrated Attention Mechanism for Speech Emotion Recognition," *IEEE Access*, 2023, doi:10.1109/ACCESS.2023.3303179. [Link to paper](https://ieeexplore.ieee.org/document/10195078)

Feel free to open issues or extend the notebooks with your own architectures and datasets.
