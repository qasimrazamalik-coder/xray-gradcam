# xray-gradcam

Fine-tunes ResNet-50 and EfficientNet-B0 to separate pneumonia from normal chest X-rays, explains each prediction with a from-scratch Grad-CAM heatmap, and reports AUROC with bootstrap intervals, sensitivity/specificity at a validation-chosen threshold, and calibration (ECE).

**Research demo. Not a medical device. Not for diagnosis.**

## Results

Test set = the official `test/` folder of the Kaggle dataset, never used for training, early stopping or threshold choice.

| Model | AUROC (95% CI) | Avg. precision | Sensitivity | Specificity | ECE |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 | not yet run | not yet run | not yet run | not yet run | not yet run |
| EfficientNet-B0 | not yet run | not yet run | not yet run | not yet run | not yet run |

Fill the table with `make data && make train && make eval`. Each run writes `results/<arch>/metrics.json` and a ready-to-paste `results/<arch>/results_table.md`. Nothing in the table is typed by hand.

## Demo

Screenshot added after first run: `docs/demo.png`.

Grad-CAM example grid and calibration plot are written to `results/<arch>/gradcam_examples.png` and `results/<arch>/reliability.png` by `make gradcam` and `make eval`.

## What question this answers

1. How much does a properly staged transfer-learning recipe (head first, then the whole network at a lower learning rate) get from ImageNet weights on a small medical dataset?
2. Do the two backbones differ once the threshold is chosen on validation data, and with confidence intervals attached?
3. Where does the model look? Grad-CAM shows whether the evidence sits inside the lungs or on borders, markers and text, which a single AUROC cannot show.
4. Are the probabilities usable as probabilities? The reliability plot and ECE answer that separately from ranking quality.

## Quickstart

```bash
git clone https://github.com/qasimrazamalik-coder/xray-gradcam.git
cd xray-gradcam
python -m venv .venv && source .venv/bin/activate
make install
make test          # unit tests, CPU, under a minute
make smoke         # train + evaluate + Grad-CAM grid on SYNTHETIC images, random weights (not a result)
make data          # Kaggle download; needs `pip install -e ".[data]"` and Kaggle credentials
make train         # both architectures, ImageNet-pretrained; needs a GPU to be practical
make eval          # test-set metrics, CIs, reliability plot
make gradcam       # correct/incorrect examples with heatmaps
make app           # Gradio demo at http://127.0.0.1:7860
```

Single architecture: `python -m xray_gradcam.train --config configs/efficientnet_b0.yaml`. The app loads `checkpoints/resnet50_best.pt` by default; use `python app/app.py --checkpoint checkpoints/efficientnet_b0_best.pt` for the other one. Without a checkpoint the app starts and shows how to train one.

## Method

**Data.** Kaggle "Chest X-Ray Images (Pneumonia)" (`paultimothymooney/chest-xray-pneumonia`), paediatric chest radiographs labelled NORMAL or PNEUMONIA, derived from the study by Kermany et al. (2018). It is class-imbalanced, with roughly three times as many pneumonia images as normal ones in `train/`. The download script prints the exact counts. **Check the licence and terms on the Kaggle dataset page before use or redistribution; this repo does not assert one.**

**Splits, and why they are re-made.** The release ships a `val/` folder of only 16 images. That is too small to drive early stopping or choose a threshold; one image moves a metric by several points. It is ignored. The official `train/` folder is re-split into train and validation (15%, stratified by label, fixed seed), and the official `test/` folder is kept untouched until `make eval`. Each results JSON stores a hash of every split so two runs can be checked for the same partition.

**Models.** torchvision ResNet-50 and EfficientNet-B0 with ImageNet weights and a new head (dropout, one logit).

**Training, the same for both architectures.**
- Stage 1: backbone frozen (BatchNorm kept in eval mode so the pretrained statistics do not drift), head trained.
- Stage 2: everything unfrozen, backbone learning rate lower than the head's, cosine schedule.
- Class imbalance: `BCEWithLogits` with `pos_weight = n_normal / n_pneumonia` (default), or a class-balanced sampler (`train.imbalance: sampler`).
- Augmentation: small rotation, shift, scale and brightness/contrast jitter. No vertical flip (not a plausible image) and no horizontal flip (it would mirror the heart to the right side).
- Mixed precision on CUDA, early stopping on validation AUROC in each stage, best weights restored.

**Threshold.** Chosen once, on the validation split, by maximising Youden's J (sensitivity + specificity - 1), stored in the checkpoint, and applied unchanged to the test set. Sensitivity and specificity therefore describe a decision rule fixed before the test data was seen.

**Metrics.** AUROC with a 95% percentile bootstrap CI (1000 resamples of test images); average precision; sensitivity, specificity and confusion matrix at the validation threshold; ECE (10 equal-width bins on the positive-class probability) with a reliability diagram.

**Grad-CAM** (Selvaraju et al., 2017), implemented in `src/xray_gradcam/gradcam.py` with forward hooks and tensor gradient hooks, no external CAM library. The target layer is named in the config: `layer4` for ResNet-50, `features.8` for EfficientNet-B0. Heatmaps use the perceptually uniform `magma` colormap rather than a rainbow map, whose bright bands suggest boundaries that are not in the data. The example grid draws correct and incorrect cases at random with a fixed seed, not by choosing attractive heatmaps.

**What keeps the comparison fair.** Same splits, seed, augmentation, loss weighting, epochs budget, threshold rule and test set for both backbones; only `model.arch` and the Grad-CAM layer differ.

**Synthetic fixture.** `data.make_synthetic_dataset` draws grayscale pictures with two dark "lungs" and, for the positive class, a brighter blob inside one lung. It is labelled SYNTHETIC in file names, plots, JSON and the app, and exists so tests and `make smoke` run on CPU in minutes. It says nothing about real performance.

## Repository structure

```
configs/    default.yaml (ResNet-50), efficientnet_b0.yaml, smoke.yaml (synthetic)
src/xray_gradcam/
  data.py       discovery, stratified split, transforms, synthetic fixture
  models.py     backbones, freezing, checkpoint IO
  train.py      two-stage fine-tuning, early stopping, validation threshold
  evaluate.py   test metrics, bootstrap CI, reliability plot, results table
  metrics.py    AUROC, AP, Youden threshold, confusion matrix, ECE, bootstrap
  gradcam.py    Grad-CAM and overlay
  examples.py   correct/incorrect Grad-CAM grid
  predict.py    batch inference and single-image explanation
app/        Gradio demo and theme
scripts/    download_data.py
tests/      pytest, CPU, synthetic data
results/    outputs of real runs (smoke output goes to results/smoke, git-ignored)
```

## Limitations

- **Possible patient-level leakage.** The dataset has no reliable patient identifiers, so the train/validation split cannot be grouped by patient. Several images from one child can land on both sides, which would inflate validation AUROC and make early stopping and the threshold optimistic. The official test folder is separate, but I cannot verify that it shares no patients with `train/` either.
- **One source population.** The data come from a single paediatric centre. Results say nothing about adults, other scanners or other hospitals. No external validation is done.
- **Threshold transfer.** The validation threshold may not carry over to the test folder if class balance or image characteristics differ between the two; compare `val` and `test` blocks in `metrics.json`.
- **Heatmaps are not proof.** Grad-CAM shows where the score is sensitive at a coarse resolution (7x7 maps at 224 px, upsampled). It can highlight borders, markers or text rather than lung tissue, and it does not establish a clinical finding. Inspect the example grid before trusting a model.
- **Bootstrap CIs** capture test-image sampling noise only, not shift, patient correlation or training randomness (single seed).
- **ECE** depends on binning and is noisy on a test set of a few hundred images.
- **Not validated clinically.** No claim is made beyond the metrics the code writes.

## Reproducing on free Colab or Kaggle

1. Runtime: GPU (T4 class is enough). On Kaggle, enable internet and add your Kaggle API token as secrets, or attach the dataset directly and set `data.root` to its path.
2. `git clone https://github.com/qasimrazamalik-coder/xray-gradcam.git && cd xray-gradcam && pip install -e ".[dev,data]"`
3. `make test && make smoke` (optional sanity check)
4. `make data && make train && make eval && make gradcam`
5. Commit `results/resnet50/` and `results/efficientnet_b0/` and paste the two `results_table.md` rows into the table above.

Runtime is approximate and depends on the assigned GPU: expect tens of minutes per model on a T4-class GPU at 224 px. I have not timed it.

## License

MIT. See `LICENSE`. The dataset keeps its own terms (see Method). Pretrained weights are torchvision's ImageNet weights.

## References

- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
- Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
- Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell, 2018.
