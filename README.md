# Brain Tumour MRI Classifier

A PyTorch classifier that labels brain MRI slices as **glioma**, **meningioma**, **pituitary tumour**, or **no tumour**. The same predictions can be collapsed into a simpler *tumour vs no tumour* detector.

> **Educational project only.** This is not a medical device and must not be used for diagnosis or any clinical decision. See [Limitations](#limitations).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/brain-tumour-mri-classifier/blob/main/notebooks/brain_tumour_mri_classifier.ipynb)

## Results

All numbers are from the held-out `Testing/` folder (1,600 images, 400 per class). They come from a single training run (seed 42) on a Colab GPU, so expect small run-to-run differences.

| Model | Input size | Test accuracy (4-class) | Tumour vs no-tumour accuracy | Tumours missed |
|---|---|---|---|---|
| Small CNN (from scratch, ~98k params) | 128 px | 89.4% | 96.4% | 58 / 1,200 |
| **ResNet18 (ImageNet-pretrained, fine-tuned)** | 224 px | **95.0%** | **98.4%** | **26 / 1,200** |

Per-class results for ResNet18:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma | 0.991 | 0.825 | 0.900 |
| meningioma | 0.891 | 0.980 | 0.933 |
| no tumour | 0.939 | 1.000 | 0.969 |
| pituitary | 0.993 | 0.995 | 0.994 |

Most remaining errors are gliomas predicted as meningioma or as "no tumour". As a detector, the model raised no false alarms on healthy scans of the tumour class (tumour precision 1.000) but missed 26 of 1,200 tumours (recall 0.978).

## Quickstart

**Google Colab (easiest):** click the badge above, choose a GPU runtime (*Runtime → Change runtime type*), and run all cells.

**Locally:**

```bash
git clone https://github.com/<your-username>/brain-tumour-mri-classifier.git
cd brain-tumour-mri-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/brain_tumour_mri_classifier.ipynb
```

The dataset downloads automatically through `kagglehub` on first run (about 150 MB). If prompted, log in to Kaggle. A GPU is strongly recommended for ResNet18 at 224 px; the small CNN is fine on a CPU.

## Choosing a model

Set `MODEL` in the config cell of the notebook:

| `MODEL` | Image size | Learning rate | Max epochs |
|---|---|---|---|
| `"resnet18"` (default) | 224 | 3e-4 | 25 |
| `"cnn"` | 128 | 1e-3 | 60 |

Early stopping (patience 6, on validation loss) usually ends training sooner. The best checkpoint is saved to `best_<MODEL>_4class.pt` and reloaded for the final test.

## Repository layout

```
brain-tumour-mri-classifier/
├── notebooks/
│   └── brain_tumour_mri_classifier.ipynb   # full pipeline: data → train → evaluate
├── docs/
│   └── METHODOLOGY.md                      # data handling, models, training, caveats
├── requirements.txt
├── LICENSE
└── README.md
```

## How it works (short version)

1. `Training/` is split 85/15 (stratified) into train and validation sets. `Testing/` is used once, at the end.
2. Images are converted to grayscale, resized, and normalised. Training images get light augmentation (flip, small shift/zoom/rotation, brightness/contrast jitter).
3. A CNN or pretrained ResNet18 outputs one score per class, trained with cross-entropy, Adam, a learning-rate scheduler, and early stopping.
4. The best checkpoint is evaluated on the test set with a classification report and confusion matrix, plus the collapsed tumour vs no-tumour view.

Details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Limitations

- **Not for clinical use.** The model was trained on a small public dataset of 2D slices and has no external validation.
- **Validation is optimistic.** Validation images come from the same folder as training images and may include near-duplicate slices from the same patients. In our run validation accuracy was 99.2% versus 95.0% on test, so treat the test figure as the honest one.
- **Possible source shortcut.** According to the dataset's description, the "no tumour" images come from a different source collection than the tumour images. The model may partly learn source or scanner differences rather than pathology, which could inflate its no-tumour performance.
- **Single run.** No confidence intervals or multiple seeds. The switch from the small CNN to ResNet18 was made after seeing the CNN's test results, so the test set is not perfectly untouched.
- **2D slices only.** No patient-level or volume-level reasoning.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar on Kaggle, which its author describes as a combination of the figshare, SARTAJ, and Br35H datasets. The version used here contains 5,600 training and 1,600 testing images across four classes. **Check the dataset page for its licence and citation requirements** before redistributing anything derived from it. The images are not included in this repository.

## License

Code is released under the [MIT License](LICENSE).
