# Speech Emotion Recognition Experiments

This repository collects experiments for speech emotion recognition using a mix of convolutional, recurrent, attention, and transformer-based models. The code is organized primarily as Jupyter notebooks so you can reproduce and extend the workflows that produced the accompanying research paper published in IEEE Access ([doi:10.1109/ACCESS.2023.3303179](https://ieeexplore.ieee.org/document/10195078)).

## Approach at a glance
The experiments are variations on the same recipe: extract stable acoustic features, model local spectral contours with convolutions, and add a sequence model to capture prosody over time. Key ingredients:

- **Acoustic front-ends**: Log-Mel spectrograms are the core representation. Transfer-learning runs reuse pretrained VGG/VGGish feature extractors for richer embeddings and faster convergence when data is scarce.
- **Temporal modeling**: CNN stacks capture local frequency patterns; LSTMs, attention pooling, and transformer blocks model longer-term temporal dependencies so the model can react to phrase-level intonation and energy contours.
- **Parallel pathways**: Several architectures process spectrograms through parallel CNN branches before fusion (e.g., transformer + attention LSTM) to capture complementary cues and make the downstream classifier more robust to noise or speaker variation.
- **Regularization & augmentation**: Noise injection, time/frequency masking, reverberation, and pitch shifts (see `dataAugmentationforSER.ipynb`) improve robustness across corpora and reduce overfitting.
- **Evaluation**: Experiments report per-epoch loss/accuracy and confusion matrices on held-out splits to highlight emotion-wise performance. Balanced accuracy is preferred when classes are uneven.

### Deeper methodological notes
- **Feature choices**: Log-Mel spectrograms with 64–128 filters and 25 ms windows work well across RAVDESS/SAVEE; MFCCs are occasionally used for ablations. Pretrained VGG/VGGish embeddings are fed directly into lightweight classifiers when compute or data are limited.
- **Architectural patterns**: Parallel CNN branches often differ in kernel sizes (e.g., {3×3, 5×5}) to cover short and mid-range spectral patterns; transformers or bidirectional LSTMs then summarize time. Attention pooling is used in notebooks labeled “attention” to focus on salient frames before classification.
- **Training details**: Adam/AdamW optimizers with learning rates between 1e-4 and 3e-4, batch sizes of 16–64, and cosine or step LR schedules are typical. Dropout (0.2–0.5) and weight decay stabilize the deeper parallel models. Early stopping on validation loss prevents overfitting on smaller corpora.
- **Evaluation protocol**: Notebooks default to train/validation/test splits within each corpus. When combining datasets, keep speaker-disjoint splits to avoid leakage. Track both overall accuracy and per-class F1; confusion matrices in the notebooks help surface emotions that systematically confuse the model (e.g., “calm” vs. “sad”).
- **Reproducibility**: Seed initialization is exposed in most notebooks; fixing seeds plus deterministic PyTorch settings yields repeatable runs. For cross-corpus comparisons, standardize audio to mono, resample to 16 kHz, and normalize amplitude before feature extraction.

## Repository contents
- **`TL_VGG_RAVDESS.ipynb`** & **`TL_VGGish_speech.ipynb`**: Transfer learning experiments that adapt VGG and VGGish feature extractors to emotion classification; good starting points when data is limited.
- **`parallel_cnn_transformer.ipynb`** & **`parallel_cnn_attention_lstm.ipynb`**: Hybrid architectures that blend convolutional front ends with transformer or attention-augmented LSTMs. Use these to test whether self-attention or recurrent attention better captures long-range prosody.
- **`stacked_cnn_lstm.ipynb`** & **`stacked_cnn_attention_lstm.ipynb`**: Deep CNN stacks feeding into recurrent layers for temporal context modeling; useful ablations for understanding how much depth helps before adding attention.
- **`cnn_LTSM.ipynb`**: Baseline convolutional + LSTM model (typo preserved from the original experiment naming) to compare against the more elaborate hybrids.
- **`dataAugmentationforSER.ipynb`**: Exploration of audio augmentation strategies to improve robustness across datasets.
- **`SAVEE Test.ipynb`** and other dataset-specific notebooks: Quick evaluations on individual corpora such as SAVEE and RAVDESS to sanity-check preprocessing.
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
4. **Verify audio assumptions**: Standardize audio to mono, 16 kHz, and similar utterance lengths; mismatches in sampling rate are a common source of silent failures.

### Recommended workflow
1. Pick a notebook that matches your modeling goal (e.g., `parallel_cnn_transformer.ipynb` for hybrid CNN-transformer or `TL_VGGish_speech.ipynb` for transfer learning).
2. Update dataset paths and sampling rates; the notebooks expect spectrogram-friendly audio (mono, consistent frame length).
3. Run the feature extraction cells first to cache spectrograms or pretrained embeddings, then train and validate. For large corpora, persist spectrograms to disk to avoid recomputation.
4. Experiment with augmentation toggles (`dataAugmentationforSER.ipynb`) to see which perturbations help the target corpus. Noise and reverberation often help cross-corpus robustness; pitch shifts help balance “happy” vs. “sad.”
5. Compare runs by reviewing printed metrics, confusion matrices, and saved checkpoints/plots; iterate on augmentation, model depth, and learning rate schedules as needed.
6. Save checkpoints for the best validation epoch. The transformer hybrids tend to peak early; deeper CNN-LSTM models sometimes need more epochs but benefit from LR warmup.

## Reproducing results
- Run the notebooks sequentially, ensuring the dataset locations and sampling rates align with your local copies. Many cells will visualize spectrograms and play back audio to aid qualitative assessment.
- Track training output (e.g., accuracy and loss) using the printed logs or by extending the notebooks to log to TensorBoard.
- For fair comparisons with the paper, fix seeds, use 16 kHz audio, and match the listed hyperparameters (filters, hidden sizes, dropout). GPU runs accelerate training, but CPU execution remains feasible for quick smoke tests.
- If you extend to new corpora, start from the transfer-learning notebooks, freeze early layers, and fine-tune classification heads; this mirrors the procedure described in the paper.

## Notes on datasets
- The notebooks were prototyped with common SER corpora such as RAVDESS and SAVEE; adapt label maps if you introduce additional corpora. RAVDESS contains acted emotions with balanced genders, while SAVEE is smaller and more sensitive to overfitting.
- Keep class balance in mind: augmentation notebooks provide recipes for balancing emotions with time/frequency masks or pitch shifts. When classes are highly imbalanced, consider class-weighted loss or focal loss experiments.
- Sampling rates vary between datasets; resample consistently before extracting spectrograms to avoid distribution shifts.
- When mixing corpora, standardize label vocabularies (e.g., map “calm” and “neutral” consistently) and inspect confusion matrices to detect label collisions.

## Citing this work
If you build on these experiments, please cite the associated paper:

> S. Ahuja , A. Shabani  "Affective Computing for Social Companion Robots Using Fine-grained Speech Emotion Recognition" [Link to paper](https://ieeexplore.ieee.org/document/10195078)

Feel free to open issues or extend the notebooks with your own architectures and datasets.
