# Liver Ultrasound Classifier — Presentation Notes

**One-liner:** Classical Image & Video Processing (IVP) turns a raw liver
ultrasound image into ~19 numeric features → a simple scikit-learn
classifier (not deep learning) predicts **Benign / Malignant / Normal**.

---

## 1. Big picture — the pipeline

```
raw image
   │
   ▼
[1] preprocessing.py   resize → grayscale → denoise → normalize → CLAHE
   │
   ▼
[2] segmentation.py    Otsu threshold + morphology → liver ROI
                        Otsu threshold + morphology → candidate lesion region
   │
   ▼
[3] feature_extraction.py   ~19 numeric features (intensity, texture, shape, edges)
   │
   ▼
[4] pipeline.py        glues [1]-[3] into ONE function used identically by
                        training AND prediction (this is the key design decision)
   │
   ├────────────────────────────┬────────────────────────────┐
   ▼                            ▼                             ▼
[5] dataset_builder.py     [6] train_model.py            [7] predict.py /
    runs [4] on the whole      trains + evaluates             liver_ui.py
    labeled dataset             4 classifiers, picks           runs [4] on ONE
    → features_dataset.npz      the best, saves models/        new image, loads
                                                                 the model, shows result
```

Say this line out loud in the viva — it's the strongest sentence in the
whole project:
> "The IVP part is *feature engineering* — turning pixels into numbers.
> The AI part is a small classical classifier that learns a decision
> boundary over those numbers. There is no CNN, no deep learning, no
> GPU — every step is a named, explainable classical algorithm."

---

## 2. File-by-file, line by line

### `preprocessing.py` — Image Preprocessing (IVP Stage 1)

| Line(s) | What happens |
|---|---|
| 25 | `TARGET_MAX_DIM = 512` — every image is normalized to a fixed size |
| 28–37 | `resize_to_target()` — resizes so the **longest side = 512px**, preserves aspect ratio, uses `INTER_AREA` when shrinking (better quality) / `INTER_LINEAR` when enlarging |
| 57 | `cv2.cvtColor(..., COLOR_BGR2GRAY)` — grayscale conversion (ultrasound is inherently intensity-based, color carries no diagnostic info) |
| 60 | `cv2.medianBlur(gray, 5)` — median filtering, first pass of **speckle noise** removal (ultrasound images have characteristic salt-and-pepper-like speckle noise from wave interference) |
| 61–63 | `cv2.fastNlMeansDenoising(...)` — **Non-Local Means denoising**, a stronger second-pass denoiser that averages similar patches across the whole image rather than just a local neighborhood (this is the slowest single operation in the whole pipeline) |
| 66 | `cv2.normalize(..., 0, 255, NORM_MINMAX)` — **intensity normalization**, stretches pixel values to fill the full 0–255 range so brightness is comparable across images from different ultrasound machines |
| 70–71 | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` — **CLAHE** (Contrast Limited Adaptive Histogram Equalization): boosts local contrast in small 8×8 tiles instead of the whole image at once, with a clip limit so noise doesn't get over-amplified (a plain global histogram equalization would blow out speckle noise) |
| 88–114 | `select_liver_roi()` — **automatic liver region detection**: Otsu-threshold the whole image → find all contours → take the **largest** contour → its bounding box = liver ROI. Falls back to the full image if the detected region is implausibly small (<10% of image area) |

**Why this order matters:** median filter → NLM denoise → normalize → CLAHE.
Denoising has to happen *before* contrast enhancement, or CLAHE would
amplify noise instead of real texture.

---

### `segmentation.py` — Region detection (IVP Stage 2)

Two separate code paths, and the docstring explicitly explains *why*:

| Lines | What |
|---|---|
| 33–67 | **Ground-truth mode** (training-dataset-only): loads human-drawn polygon JSON annotations, scales them to match the resized image, converts to OpenCV contour format. *Not actually used in the final pipeline* — kept for reference/earlier calibration. |
| 74–92 | `detect_candidate_regions()` — the automatic path used everywhere now: <br>• `GaussianBlur` (5×5) to smooth before thresholding <br>• `cv2.threshold(..., THRESH_BINARY_INV + THRESH_OTSU)` — **Otsu's method** automatically picks the optimal binary threshold from the histogram (no manual tuning) <br>• `MORPH_OPEN` then `MORPH_CLOSE` with an elliptical kernel — **morphological opening** removes small noise specks, **closing** fills small holes/merges nearby fragments <br>• `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` — extracts outer contours only, sorted largest→smallest |
| 95–169 | `best_candidate_region()` — doesn't just take the biggest blob; **scores every candidate contour** on a weighted formula (lines 157–163): <br> `0.30·circularity + 0.25·solidity + 0.20·extent + 0.15·contrast + 0.10·area_score` <br>This is a hand-designed "does this look like a lesion" heuristic — circular, solid (not a thin sliver), reasonably filled bounding box, and different in intensity from surrounding liver tissue. |

**Shape descriptors explained (good viva material):**
- **Circularity** = `4π·Area / Perimeter²` → 1.0 for a perfect circle, lower for irregular/elongated shapes
- **Solidity** = `Area / ConvexHullArea` → how "filled in" vs. how ragged/concave the boundary is
- **Extent** = `Area / BoundingBoxArea` → how much of its own bounding box the region fills

---

### `feature_extraction.py` — turning a region into numbers (IVP Stage 3)

- **Lines 37–63** — `FEATURE_NAMES`: the fixed, ordered list of **19 features**. Order matters everywhere downstream (training and prediction must agree exactly).
- **Lines 70–130** — `_glcm_features()`: computes the **Gray-Level Co-occurrence Matrix (GLCM)**, a classic texture-analysis technique.
  - Line 94–96: quantizes 256 gray levels down to 32 (keeps the co-occurrence matrix small/fast)
  - Line 104–116: `skimage.feature.graycomatrix` — counts how often pairs of pixel intensities occur next to each other, at 4 angles (0°, 45°, 90°, 135°) and distance 1, then **averages across all 4 directions** so the result is rotation-invariant
  - Line 118–128: `graycoprops` extracts 3 standard GLCM statistics:
    - **Contrast** — how much local intensity varies (high = coarse/irregular texture)
    - **Energy** — uniformity (high = few dominant gray levels, smooth/homogeneous)
    - **Homogeneity** — closeness of the distribution to the diagonal (high = smooth, low = textured)
- **Lines 137–161** — `_edge_density()`: **Canny edge detection** (thresholds 40/120) → edge density = edge pixel count ÷ total pixels. A proxy for how irregular/structured the region's boundary and internal texture is.
- **Lines 168–252** — `extract_liver_features()`: runs mean/std intensity + GLCM + edge density over the **whole liver ROI** (background/reference statistics).
- **Lines 259–565** — `extract_mass_features()`: same statistics but for the **candidate lesion only**, plus shape descriptors (circularity, extent, solidity, aspect ratio) and — importantly — **lines 421–477**: masks out everything *outside* the lesion contour before computing GLCM/edges, so background liver tissue doesn't contaminate the lesion's own texture signature.
- **Lines 572–607** — `empty_mass_features()`: if no candidate region was found at all, returns an all-zero feature block with `mass_detected = 0` — this flag itself becomes a learned signal.
- **Lines 614–627** — `features_dict_to_vector()`: dict → fixed-order NumPy array (this is literally one row of the training matrix / one prediction input).

---

### `pipeline.py` — the single source of truth (glue layer)

This is the most *conceptually* important file, even though it's short.

- **Lines 42–75**, `process_bgr()`: liver ROI → liver features → candidate region → mass features → merge dict → vector. Returns everything (`liver_bbox`, `mass_bbox`, `feature_vector`) needed by both training and the UI.
- **Lines 78–83**, `process_image_path()`: file path → preprocess → `process_bgr()`.
- **The docstring (lines 6–27) is the whole story of a bug that was found and fixed** — see Section 4, this is excellent viva material.

---

### `dataset_builder.py` — builds the training matrix

- Lines 32–33: expects `Benign/`, `Malignant/`, `Normal/` folders (each with an `image/` subfolder).
- Lines 36–64, `build_dataset()`: loops every image in every class folder → runs `pipeline.process_image_path()` (the **exact same function** used at prediction time) → appends `feature_vector` + class label.
- Line 90: `np.savez_compressed()` saves `X` (features), `y` (labels), and `feature_names` into `features_dataset.npz` so this slow step (~5–8 min for 735 images) only has to run once.

### `train_model.py` — model selection & training (the "AI")

- Lines 90–129, `create_candidate_models()`: 4 classical classifiers compared head-to-head — **Logistic Regression, SVM (RBF kernel), Random Forest, K-Nearest Neighbors**.
- Lines 136–188, `cross_validate_models()`: **5-fold stratified cross-validation** (`StratifiedKFold` keeps class proportions equal in every fold) — picks the best baseline by mean CV accuracy.
- Lines 195–263, `tune_svm()`: separate **grid search** over SVM's `C` (regularization strength) and `gamma` (kernel width) hyperparameters.
- Lines 546–558: **`StandardScaler` fit only on the training split**, then applied to test — explicit comment about avoiding data leakage.
- Lines 590–616: whichever is better — tuned SVM or the best untouched baseline — is selected as `best_model`.
- Lines 661–671: **`CalibratedClassifierCV`** (Platt/sigmoid scaling) wraps the final model so `predict_proba()` gives *meaningful* probabilities, not just raw decision scores — calibration doesn't change *accuracy*, only how trustworthy the confidence percentages are.
- Lines 696–757: saves `liver_classifier.joblib`, `feature_scaler.joblib`, `feature_names.joblib`, `model_name.joblib`, `training_metadata.joblib` into `models/`.

### `predict.py` — CLI inference on new images

- Lines 45–69, `analyze_image()`: loads one image → `process_image_path()` → scale → `model.predict()` / `predict_proba()`.
- Lines 72–90, `annotate_result()`: draws the liver ROI box (gray) and lesion box (colored by predicted class) with OpenCV `rectangle`/`putText` onto a copy of the resized image.
- Lines 93–111: prints a formatted terminal report, including an ASCII probability bar chart and an explicit "not a medical diagnosis" disclaimer.

### `liver_ui.py` — the desktop GUI (Tkinter)

- Built with `tkinter`/`ttk` — a single-window, two-column dashboard (input image + annotated output on the left, verdict/probabilities/feature table on the right).
- `analyze()` calls the same `pipeline.process_image_path()` as the CLI — **the UI never re-implements the IVP logic**, it just visualizes the same pipeline's output.
- Runs the pipeline on a background `threading.Thread` (not IVP-related — this was a responsiveness fix so the window doesn't freeze during the ~1s of OpenCV work) and hands results back to the main thread via `root.after()`.
- `_annotate_result()`, `_update_verdict()`, `_update_probabilities()` mirror `predict.py`'s CLI report as GUI widgets (progress bars for class probabilities, a feature table, colored verdict text).

---

## 3. IVP (Image & Video Processing) concepts used — full list

| Concept | Where (file : line) | Why it's used |
|---|---|---|
| Image resizing / scale normalization | `preprocessing.py:28-37` | Consistent feature scale regardless of source resolution |
| Grayscale conversion | `preprocessing.py:57` | Ultrasound is intensity-based; no color info to lose |
| Median filtering | `preprocessing.py:60` | First-pass speckle noise removal |
| Non-Local Means denoising | `preprocessing.py:61-63` | Stronger, edge-preserving speckle noise removal |
| Intensity normalization (min-max) | `preprocessing.py:66` | Consistent brightness range across machines |
| CLAHE (adaptive histogram equalization) | `preprocessing.py:70-71` | Local contrast boost without over-amplifying noise |
| Otsu's thresholding | `preprocessing.py:100-101`, `segmentation.py:82-83` | Automatic, data-driven binarization (no manual threshold) |
| Morphological opening / closing | `segmentation.py:85-87` | Remove small noise blobs / merge nearby fragments |
| Contour detection & largest-contour selection | `preprocessing.py:102-109`, `segmentation.py:89-92` | Region boundary extraction for ROI selection |
| Convex hull / shape descriptors | `segmentation.py:122-134`, `feature_extraction.py:304-347` | Circularity, solidity, extent, aspect ratio — "does this look like a lesion" |
| GLCM texture analysis | `feature_extraction.py:70-130` | Contrast, energy, homogeneity — classic texture quantification |
| Canny edge detection | `feature_extraction.py:137-161` | Edge density as a structure/margin signal |
| Region masking | `feature_extraction.py:352-368, 424-444` | Isolate exact lesion pixels from surrounding tissue for accurate stats |

**No deep learning, no CNNs, no learned segmentation network anywhere** — every one of the above is a classical, explainable, textbook IVP algorithm. Worth saying explicitly to faculty.

---

## 4. AI / ML concepts used

| Concept | Where | Notes |
|---|---|---|
| Feature-based classification (not raw pixels) | whole project | Model sees ~19 engineered numbers per image, not the image itself |
| Train/test split | `train_model.py:518-526` | 80/20, **stratified** (keeps class ratios equal in both splits) |
| Feature standardization | `train_model.py:546-558` | `StandardScaler`, fit on train only — avoids data leakage into test |
| K-fold cross-validation | `train_model.py:136-188` | 5-fold stratified, used for fair model comparison |
| Model comparison | `train_model.py:90-129` | Logistic Regression vs SVM(RBF) vs Random Forest vs KNN |
| Hyperparameter tuning (grid search) | `train_model.py:195-263` | `GridSearchCV` over SVM's `C` and `gamma` |
| Class imbalance handling | `train_model.py` (`class_weight="balanced"`) | Dataset is imbalanced: 435 Malignant vs 100 Normal |
| Probability calibration | `train_model.py:661-671` | `CalibratedClassifierCV` (sigmoid/Platt scaling) |
| Evaluation metrics | `train_model.py:270-364` | Accuracy, macro/weighted F1, confusion matrix, per-class precision/recall |
| Feature importance | `train_model.py` (Random Forest branch) | Which of the 19 features actually drives predictions |

---

## 5. If asked "how would you improve this?"

- Tighten `best_candidate_region()`'s scoring/thresholds to cut down false-positive candidates on Normal images (the single biggest lever, per README)
- Grid-search Random Forest's `max_depth` / `n_estimators` more thoroughly
- Add more IVP features (Laplacian variance for blur, more GLCM angles/distances, Local Binary Patterns) — just extend `FEATURE_NAMES` + the extraction function, pipeline stays unchanged
- Address class imbalance further (currently only `class_weight="balanced"`) — could add oversampling (e.g. SMOTE) for Normal/Benign

## 6. Likely faculty questions & quick answers

- **"Why not a CNN?"** → Small dataset (735 images), no GPU requirement, fully explainable feature-by-feature, trains in seconds — appropriate for the IVP syllabus where the algorithms themselves are the point.
- **"Why Otsu instead of a fixed threshold?"** → Otsu picks the threshold automatically from each image's own histogram (minimizes intra-class variance) — robust to brightness differences between images/machines instead of needing manual re-tuning per image.
- **"Why GLCM and not just intensity mean/std?"** → Mean/std only capture *how bright*, GLCM captures *spatial texture patterns* (smooth vs. coarse vs. structured) — malignant tissue tends to have different texture, not just different brightness.
- **"What does `mass_detected = 0` mean and why keep it as a feature?"** → It tells the model "no candidate region was found at all" — itself informative, since Normal images more often have no detectable candidate.
- **"How do you prevent data leakage?"** → `StandardScaler` fit only on the training split (`train_model.py:546-558`); stratified train/test split with a fixed `random_state`.
