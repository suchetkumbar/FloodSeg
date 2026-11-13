# Dataset & Model Citations and Licenses

This project adheres to ethical open-source machine learning practices. All datasets, foundation models, and core libraries used in FloodSeg are documented below with their official academic citations and open-source licenses.

---

## 1. Sen1Floods11 Dataset

* **Citation:**
  Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 210–211.
* **BibTeX:**
  ```bibtex
  @inproceedings{bonafilia2020sen1floods11,
    title={Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1},
    author={Bonafilia, Derrick and Tellman, Beth and Anderson, Tyler and Issenberg, Erica},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    pages={210--211},
    year={2020}
  }
  ```
* **License:** Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
* **Repository & Data Access:** [Cloud to Street Sen1Floods11 GitHub](https://github.com/cloudtostreet/Sen1Floods11) & Google Cloud Storage bucket `gs://sen1floods11/`.

---

## 2. Satellite Data Providers (Copernicus Sentinel Missions)

* **Sentinel-1 SAR:** European Space Agency (ESA) Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) C-Band imagery.
* **Sentinel-2 Multispectral:** European Space Agency (ESA) Copernicus Sentinel-2 MultiSpectral Instrument (MSI) Level-1C / Level-2A.
* **License:** Open Access policy governed by the European Union Copernicus program (free, full, and open access to Sentinel data).
* **Case Study Access Provider:** Microsoft Planetary Computer STAC API (Open Access).

---

## 3. Deep Learning Frameworks & Pretrained Backbones

### PyTorch & Torchvision
* **Citation:**
  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An imperative style, high-performance deep learning library.* Advances in Neural Information Processing Systems (NeurIPS), 32.
* **License:** Modified BSD License.

### Segmentation Models PyTorch (SMP)
* **Citation:**
  Iakubovskii, P. (2019). *Segmentation Models Pytorch.* GitHub repository, `https://github.com/qubvel/segmentation_models.pytorch`.
* **License:** MIT License.

### ResNet-34 Encoder (ImageNet Pretrained)
* **Citation:**
  He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 770–778.
* **Pretrained Weights License:** Open-source research weights (Torchvision / PyTorch model zoo, BSD 3-Clause).

---

## 4. Scientific Python Ecosystem

* **Rasterio:** Mapbox / Sean Gillies et al., BSD 3-Clause License.
* **Albumentations:** Buslaev et al., MIT License.
* **Hydra:** Omry Yadan / Meta, MIT License.
* **Weights & Biases (W&B):** MIT License (Python SDK) / Free academic & individual tier.

---

## 5. Methods

* **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015, LNCS 9351, pp. 234–241.
* **NDWI:** McFeeters, S. K. (1996). *The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.* International Journal of Remote Sensing, 17(7), 1425–1432.
* **MNDWI:** Xu, H. (2006). *Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery.* International Journal of Remote Sensing, 27(14), 3025–3033.
* **Otsu thresholding:** Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.
* **Calibration (ECE):** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML 2017.

---

## 6. Case Study Data (Milestone 5)

* **Sentinel-1 RTC (Planetary Computer):** radiometrically terrain-corrected Sentinel-1 GRD backscatter (gamma0), produced by Catalyst/PCI and hosted by Microsoft Planetary Computer, collection `sentinel-1-rtc`. Contains modified Copernicus Sentinel data; see the collection page for licence terms.
* **Sentinel-2 L2A (Planetary Computer):** collection `sentinel-2-l2a`, Copernicus open-access licence.
* **ESA WorldCover 2021 v200:** Zanaga, D., Van De Kerchove, R., Daems, D., et al. (2022). *ESA WorldCover 10 m 2021 v200.* Zenodo. https://doi.org/10.5281/zenodo.7254221. Licence: CC BY 4.0. Used for the permanent-water mask (class 80).
* **Microsoft Planetary Computer:** Microsoft Open Source, McFarland, M., Emanuele, R., Morris, D., & Augspurger, T. (2022). *microsoft/PlanetaryComputer.* Zenodo. https://doi.org/10.5281/zenodo.7261897
* **Basemaps (interactive map):** © OpenStreetMap contributors (ODbL); Esri World Imagery (Esri, Maxar, Earthstar Geographics, and the GIS User Community).
* **Libraries:** pystac-client (Apache 2.0), planetary-computer (MIT), Folium (MIT), SciPy (BSD 3-Clause).
