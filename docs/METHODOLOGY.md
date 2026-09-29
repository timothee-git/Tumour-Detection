# Methodology

This document explains what the notebook does and why, and how much weight the results can bear.

## 1. Data

- **Source:** Kaggle "Brain Tumor MRI Dataset", downloaded with `kagglehub`.
- **Layout:** `Training/` and `Testing/` folders, each with four class sub-folders (`glioma`, `meningioma`, `notumor`, `pituitary`). `torchvision.datasets.ImageFolder` turns folder names into labels.
- **Sizes (as downloaded):** 5,600 training images (1,400 per class) and 1,600 testing images (400 per class), so classes are balanced and no class weighting is used.
- **Validation split:** 15% of `Training/` (840 images) is held out, stratified by class, with a fixed seed. This leaves 4,760 images for training.
- **Test set:** `Testing/` is used only for the final evaluation.

Two `ImageFolder` views of the training folder are created, one with augmentation (for training) and one without (for validation), then indexed by the same split.

## 2. Preprocessing and augmentation

| Step | Train | Validation / Test |
|---|---|---|
| Grayscale (1 channel) | yes | yes |
| Resize to `IMG_SIZE` × `IMG_SIZE` | yes | yes |
| Random horizontal flip | yes | no |
| Random affine (±10° rotation, up to 5% shift, 0.9–1.1 zoom) | yes | no |
| Brightness / contrast jitter (±20%) | yes | no |
| Normalise to mean 0.5, std 0.5 | yes | yes |

## 3. Models

**Small CNN** (`MODEL = "cnn"`, ~98k parameters): four blocks of Conv 3×3 → BatchNorm → ReLU → MaxPool with 16, 32, 64, 128 channels, then global average pooling, dropout (0.3), and a linear layer with 4 outputs. Global pooling means the classifier does not depend on input size.

**ResNet18** (`MODEL = "resnet18"`, ~11.2M parameters): torchvision's ImageNet-pretrained ResNet18. The first convolution is replaced by a 1-channel version whose weights are the pretrained RGB weights summed over the colour axis, so grayscale input works without changing the pipeline. The final layer is replaced with dropout (0.3) and a 4-way linear layer. All layers are fine-tuned.

Both models output raw logits.

## 4. Training

- Loss: `CrossEntropyLoss`
- Optimiser: Adam, weight decay 1e-4, learning rate per model (see README)
- Scheduler: `ReduceLROnPlateau` on validation loss (factor 0.5, patience 2)
- Early stopping: stop after 6 epochs without a new lowest validation loss
- Checkpoint: the weights with the lowest validation loss are saved and reloaded for testing
- Seed: 42 (GPU non-determinism means exact numbers can still vary)

The notebook prints per-epoch train/validation loss and accuracy, plots both curves with the saved-checkpoint epoch marked, and runs a "plateau check" that reads the history and suggests what to change (more epochs, bigger model, more regularisation, or "converged").

## 5. Evaluation

Run once on `Testing/` after training: a per-class precision/recall/F1 report and a confusion matrix. A second view collapses predictions to *tumour vs no tumour* (any of the three tumour classes counts as "tumour") and reports how many tumours were called "no tumour", the costliest error for a detector.

## 6. Development history

1. **Baseline:** the original notebook trained on binary labels, used the test folder for validation, ran 600 epochs, and had ordering and labelling bugs. It was rebuilt with a proper train/validation/test flow and a modern loss.
2. **Four-class version:** relabelling was removed so the model predicts all four classes; the tumour vs no-tumour score is derived afterwards.
3. **Small CNN result:** 89.4% test accuracy. The training curves showed underfitting (train accuracy below validation, still improving at epoch 40).
4. **ResNet18 + stronger augmentation:** 95.0% test accuracy, with early stopping at epoch 16 of a possible 25.

## 7. How much to trust the numbers

- Validation accuracy (99.2%) is much higher than test accuracy (95.0%). This is consistent with near-duplicate slices across the train/validation split and/or a difference in distribution between the two folders; the run itself cannot tell which.
- The dataset provides no patient IDs, so a patient-level split is not possible.
- The model choice was made after seeing test results for the small CNN, so the reported test score carries mild selection bias.
- One seed, one run: differences of a point or two between configurations should not be over-interpreted.
- Nothing here has been validated on data from other hospitals or scanners.

## 8. Ideas for future work

- Group-aware or external validation to replace the optimistic validation split
- Multiple seeds with confidence intervals
- Larger backbones (ResNet34/50, EfficientNet) and test-time flip averaging
- Visual explanations (e.g. Grad-CAM) to check the model looks at the tumour rather than image borders or source artefacts
