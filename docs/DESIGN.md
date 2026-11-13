# FloodSeg: Architectural & System Design Document (DESIGN.md)

## 1. Problem Framing & Core Objective
Flood extent mapping from satellite imagery during disaster response presents a fundamental trade-off:
* **Optical Sensors (Sentinel-2):** Provide rich multispectral signatures (NDWI/MNDWI from Green, NIR, SWIR) to unambiguously detect surface water, but are rendered completely opaque by heavy cloud cover and severe storm systems accompanying monsoon floods.
* **Synthetic Aperture Radar (Sentinel-1 SAR):** Active C-band radar penetrates cloud cover, precipitation, and darkness. Smooth open water acts as a specular reflector, scattering radar pulses away from the sensor and appearing dark (low backscatter in VV/VH). However, radar suffers from speckle noise, geometric distortion, and ambiguity with smooth radar-dark surfaces (such as flat airport runways, asphalt, or dry smooth soil), as well as vegetative canopy obstruction.
* **Core Question:** *How much does SAR improve flood mapping over optical imagery under adverse atmospheric conditions, and where does each modality fail?*

---

## 2. Data Integrity & Spatial Leakage Prevention

### 2.1 The Spatial Autocorrelation Problem
In Earth observation and remote sensing, spatial autocorrelation (Tobler's First Law of Geography: *"everything is related to everything else, but near things are more related than distant things"*) presents a major trap in benchmark evaluation.
* **The Random Chip Leakage Failure Mode:**
  If $512 \times 512$ chips cut from the same satellite acquisition or neighboring geographic tiles of the same flood event are randomly assigned across train and test splits:
  1. The model memorizes regional soil moisture signatures, local land cover distributions, and specific hydrological geometries.
  2. Test set metrics (IoU, F1) become artificially inflated because the evaluation partition shares nearly identical radiometric distributions and spatial features with the training set.
  3. The model appears to generalize brilliantly on the test set, but collapses when deployed onto an unseen flood event in another geographic basin.

### 2.2 Dual-Split Protocol
To ensure rigorous evaluation, FloodSeg implements two distinct split regimes:
1. **Official Split:** The standard Sen1Floods11 partition (252 train, 89 val, 90 test chips), allowing direct, apples-to-apples comparison against published research benchmarks (Bonafilia et al., 2020).
2. **Leave-One-Event-Out (LOEO) Split:** Entire flood events (e.g. holding out the India Assam event and Bolivia event completely) are excluded from training and validation, and used exclusively for out-of-distribution evaluation. This provides the ground-truth measure of real-world operational generalization.

### 2.3 No-Data / Invalid Masking
Sen1Floods11 labels contain three distinct classes:
* `1`: Surface / Flood Water
* `0`: Land / Non-water
* `-1`: No-Data / Invalid / Cloud artifact
**Strict Invariant:** Pixels with value `-1` are explicitly masked out from all loss computations (using PyTorch `ignore_index = -1` or binary reduction masks) and excluded from confusion matrices. Computing accuracy over invalid pixels would introduce severe systematic bias.

---

## 3. Data Normalization & Augmentation Policy

### 3.1 Strict Train-Only Normalization
* Normalization statistics (per-band mean and standard deviation) are computed strictly across the **training split** of the active split regime.
* Validation, test, and out-of-distribution chips are standardized using the pre-computed training statistics, preventing test-set data leakage.
* Statistics are stored **per split regime** (`data/stats/train_stats_{official,loeo}_{s1,s2,fusion}.json`). The official train split contains 40 India chips, so LOEO runs must use LOEO-train statistics; otherwise the held-out India event leaks into preprocessing.
* Non-finite values (SAR NaNs) are excluded per channel, not per pixel.

### 3.2 Physics-Aware Augmentation Rules
Data augmentation must respect the physical properties of the sensors:
* **Allowed Spatial Augmentations:**
  * Horizontal and vertical flips
  * Random 90° rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)
  * Random resized crops (preserving aspect ratio)
* **Forbidden Augmentations:**
  * **Color Jitter / Hue / Saturation Shifts on SAR:** Sentinel-1 VV and VH backscatter channels represent calibrated physical power ratios (dB, decibels). Applying random photometric jitter, contrast stretching, or hue shifts destroys the radiometric calibration and alters the physical reflection thresholds.
  * Gaussian noise on raw SAR is avoided because SAR already contains multiplicative Rayleigh/speckle noise.

---

## 4. Modeling & Fusion Architecture

### 4.1 Backbone & Encoder Adaptation
* **Base Architecture:** U-Net with a ResNet-34 encoder initialized with ImageNet weights.
* **Pretrained Weight Transfer for $N$-Band Inputs:**
  Standard ResNet architectures expect 3 RGB input channels. For multispectral ($N=13$ or $N=6$) or fused ($N=15$) inputs:
  1. We preserve the pretrained weights for the RGB channels (Sentinel-2 B4, B3, B2).
  2. For the extra $(N-3)$ channels, we initialize kernel weights using the channel-averaged mean of the pretrained RGB weights:
     $$W_{\text{extra}} = \frac{1}{3} \sum_{c=1}^3 W_{\text{pretrained}}[:, c, :, :]$$
  3. We apply an input variance scaling factor $\frac{3}{N}$ across the entire first layer kernel tensor to preserve the expected activation variance entering the residual stages.

### 4.2 Fusion Strategies
1. **Early Fusion (Channel Concatenation):**
   * Sentinel-1 (VV, VH) and Sentinel-2 bands are stacked along the channel dimension at the input level $(C = 2 + 13 = 15)$.
   * *Advantage:* Low computational overhead; the network learns cross-modal pixel correlations directly in the earliest convolutional layers.
2. **Late Fusion (Dual Encoder - Stretch Goal):**
   * Independent encoder streams for SAR and Optical, fused at the bottleneck or multi-scale skip connections.

---

## 5. Loss Function & Class Imbalance

### 5.1 Imbalance Dynamics
Flood water typically occupies between 5% and 25% of any given satellite scene. Standard Binary Cross-Entropy (BCE) trained on such distributions is easily overwhelmed by the dominant negative class (land), leading to conservative models with high precision but low recall.

### 5.2 Combined Loss Formulation
We employ a combined loss:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{Dice}}$$
where $\alpha = 0.5$, $\beta = 0.5$.
* **Soft Dice Loss:** Directly optimizes the spatial overlap (Sørensen–Dice coefficient / IoU surrogate), insensitive to background class prevalence.
* **BCE Loss:** Provides smooth, convex gradient landscapes during early training stages to prevent local minima.
* **Ablation Study:** A dedicated experiment tests Dice-only vs. BCE+Dice to validate whether joint optimization delivers superior boundary calibration.

---

## 6. Baselines & Evaluation Metrics
Non-learned baselines are mandatory reference points:
1. **Optical Baseline:** Normalized Difference Water Index (NDWI = $\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$) and Modified NDWI (MNDWI = $\frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}}$). Threshold tuned on the train set only via grid search.
2. **SAR Baseline:** Otsu automated bimodal thresholding on VV backscatter (dB).
3. **Core Metric:** **Intersection over Union (IoU) of the Water class**, accompanied by F1-Score, Precision, Recall, and Expected Calibration Error (ECE) with bootstrap 95% confidence intervals.

### 6.1 Aggregation & Confidence Intervals
* **Pooled (micro) IoU** is the headline: confusion counts are summed over every valid pixel in the split, then IoU is computed once. **Chip-mean (macro) IoU** is reported alongside it because it weights small floods equally.
* CIs resample **chips** (the independent unit; pixels within a chip are strongly autocorrelated) and recompute the *same* aggregate, so each CI brackets its own point estimate.

### 6.2 Threshold Tuning Protocol
* Baseline thresholds maximise pooled train IoU over a fixed grid (indices: step 0.01 on [-0.5, 0.5]; VV: step 0.1 dB on [-25, -8]).
* Thresholds are tuned **separately per split regime** on that regime's train split. Tuning LOEO baselines on the official train split would leak India chips into the held-out evaluation.

---

## 7. Training Protocol (Milestone 3)

* **Grid:** {S1, S2, fusion} x {official, LOEO} x seeds {42, 1337, 2024} with BCE+Dice (18 runs), plus a Dice-only ablation (official, seed 42, 3 runs). Driver: `scripts/run_m3_grid.py`; per-run config is stored by Hydra in `outputs/runs/<run>/.hydra/`.
* **Inputs:** random 256x256 crops for training (as in Bonafilia et al., 2020), followed by flips and 90° rotations; validation and test use full 512x512 tiles.
* **Optimisation:** AdamW (lr 3e-4, wd 1e-4), batch 16, 3-epoch linear warmup then cosine decay to 1e-6, gradient clipping at 1.0, AMP (forward FP16, loss FP32), at most 50 epochs.
* **Model selection:** best checkpoint and early stopping (patience 12) on *validation* pooled water IoU. The test split (official test, or LOEO held-out India + Bolivia) is evaluated exactly once per run, with the best checkpoint. Official-regime models are also scored on the official Bolivia hold-out.
* **Data cache:** decoded chips are cached as channel-first float16 `.npy` (<0.05% relative error, far below sensor noise) and memory-mapped, so a crop reads only its own window and an S1 run reads only 2 of the 15 bands. Without the cache, epochs were bound by GeoTIFF decoding and disk throughput (~16 s vs ~7 s per epoch).
* **Determinism:** all seeds are fixed, but cuDNN autotuning is enabled (`deterministic: false`) for speed, so runs are not bit-identical. Seed-to-seed spread is reported explicitly.
* **Statistics:** seed mean ± sample std of pooled test IoU. Modality differences use a *paired* chip-level bootstrap on seed-summed per-chip confusion counts (`scripts/aggregate_m3.py`). Pairing on identical chips removes between-chip variance.

---

## 8. Failure Analysis & Cloud Robustness (Milestone 4)

### 8.1 Why synthetic clouds
Sen1Floods11 hand labels mark clouded pixels as no-data (`-1`), so the benchmark only scores pixels the optical sensor could see. That systematically favours optical inputs and cannot answer the headline question. We therefore **occlude the optical bands of labelled pixels** and keep the true label.

### 8.2 Cloud model (`floodseg.data.clouds`)
* **Spectrum:** median L1C TOA reflectance of thick-cloud pixels (B2, B3, B4 > 0.35) in the official *train* split (93 chips, 30,584 sampled pixels). It is spectrally flat in the visible, bright in NIR and SWIR1, and dark in cirrus B10, as real low cloud is.
* **Shape:** a Gaussian random field (σ ≈ 24 px ≈ 240 m), thresholded at its (1 − coverage) quantile. A soft ramp gives semi-transparent edges, so coverage is exact in expectation. Fields are generated at ¼ resolution and bilinearly upsampled (visually identical, ~16× cheaper).
* **Texture:** ±20% smooth brightness modulation.
* **Physics:** only S2 channels are modified; SAR passes through unchanged (C-band penetrates cloud). Clouds are blended into raw values *before* normalisation, exactly like real data.
* **Evaluation clouds** are deterministic per (chip, coverage) via a CRC32-seeded RNG, so every model sees identical clouds and differences are paired.

### 8.3 Variants compared
1. **Standard models** trained on clear-sky data.
2. **Perfect cloud mask:** pixels with opacity ≥ 0.5 have their S2 values replaced by the train mean (i.e. 0 after normalisation). This is an upper bound on what a cloud-masking pre-processor achieves without retraining.
3. **Cloud-augmented training** (`train.synthetic_cloud_p = 0.5`, coverage ~ U(0, 1)): models see occluded optical bands during training, so fusion can learn to fall back on SAR. Training clouds use torch's per-worker RNG and never coincide with the deterministic evaluation fields.

### 8.4 Real-chip stratification
Test chips are binned by (a) a heuristic cloud proxy (B2, B3, B4 > 0.18 or B10 > 0.012) and (b) the labelled no-data fraction. The proxy agrees only moderately with the labellers' mask (P(no-data | proxy) = 0.40, P(proxy | no-data) = 0.58), so it is used for stratification only. Because clouded pixels carry no label, these bins measure performance on the *visible* part of cloudy scenes (adjacency effects, thin cloud, haze, shadow).

### 8.5 Statistics
Per-model numbers are pooled IoU, mean ± s.d. over seeds. Differences between models use per-chip confusion counts summed over seeds and a 5,000-resample **paired** chip bootstrap.

---

## 9. Real-World Case Study (Milestone 5)

* **Area and grid:** Kuttanad / Alappuzha (76.28–76.58°E, 9.28–9.62°N), the most severely inundated low-lying region of the August 2018 Kerala floods. Data are reprojected onto Sen1Floods11's own grid (EPSG:4326, 8.98 × 10⁻⁵°, ~10 m) so the model sees its training geometry.
* **Inputs:** Sentinel-1 RTC from Microsoft Planetary Computer (STAC), descending relative orbit 165:
  * flood scene 2018-08-21, two frames mosaicked;
  * dry-season reference: per-pixel median of 2018-02-22, 03-06 and 03-18 (a temporal median suppresses speckle).
  * Linear gamma0 is converted to dB.
* **Model:** the SAR-only U-Net, 3-seed ensemble (official regime), via the same tiled inference as `floodseg predict` (512 px tiles, 128 px overlap, Hann blending).
* **Flood extent:** water on the flood date minus reference water. Reference water = dry-season SAR water ∪ ESA WorldCover 2021 class 80 (permanent water). The WorldCover term is required: wind-roughened sea is radar-bright in some reference scenes, and without it about 86 km² of open sea would be reported as flood (400 vs 314 km²).
* **Independent check:** Sentinel-2 L2A of 2018-08-22 (one day later, 31% tile cloud cover). On pixels the Scene Classification Layer marks clear (vegetation, bare, water, unclassified), SAR water is compared with MNDWI > 0.15 and with the SCL water class.
* **Known limitations:**
  * RTC gamma0 differs from the GEE sigma0 used in training (scene VV mean −12.7 dB vs −10.4 dB); no recalibration is applied.
  * The S1 pass is ~4 days after the peak.
  * Optical agreement is not ground truth.
