# AnnotateAnyCell

Open-source semi-supervised deep learning framework for efficient annotation and analysis of whole slide images in digital pathology.

[![Paper](https://img.shields.io/badge/Paper-Available-blue)](https://www.biorxiv.org/content/10.1101/2025.11.02.686114v1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

AnnotateAnyCell is an intelligent interactive web-based platform that enables rapid annotation of cellular structures in histopathological images. The framework integrates active contrastive learning with dimensionality reduction techniques to facilitate efficient exploration and labeling of cells at scale.

**Key Capabilities:**
- Interactive annotation in learned embedding space using UMAP visualization
- Active learning mechanism that improves with human-in-the-loop feedback
- Multi-modal cellular representation (raw patches, isolated regions, semantic masks)
- Real-time model retraining with live progress updates
- Quantitative feature extraction and analysis at cellular resolution

---

## Architecture

The framework implements a four-stage iterative pipeline:

1. **Preprocessing:** Cell and nuclear segmentation using Cellpose models, generating multi-modal representations (128×128 tiles for training, 512×512 for inspection)

2. **Embedding Generation:** Dimensionality reduction via UMAP to visualize morphologically similar cells in 2D space

3. **Active Annotation:** Expert pathologists label cells within the embedding space, with intelligent sample selection prioritizing informative examples

4. **Model Refinement:** Contrastive learning framework generates pseudolabels and updates embeddings, progressively improving accuracy

---

## Interface

### Annotation Workspace

![Main Annotation Interface](main_ui.png)

The annotation interface features three coordinated panels:
- **Left:** Workflow guide with nuclear size and N/C ratio distributions
- **Center:** Interactive UMAP embedding where cells cluster by morphological similarity
- **Right:** High-resolution cell preview with morphological annotation controls

### Results and Analysis

![Output Interface](output_ui.png)

The output interface provides:
- Annotation summary statistics and distribution analysis
- Whole slide visualization with spatial location markers
- Individual cell inspection and detailed morphological classifications
- CSV export for downstream computational analysis

---

## Applications

- Accelerated annotation of large-scale histopathological datasets
- Cancer biology research requiring cellular phenotype quantification
- Personalized medicine studies analyzing tissue heterogeneity
- Clinical pathology workflows for diagnostic feature extraction

---

## Technical Details

**Segmentation:** Pretrained Cellpose models (cyto3 for cells, nuclei for nuclear structures)

**Feature Extraction:** Multi-modal representation learning with convolutional autoencoder

**Embedding:** UMAP-based dimensionality reduction for interactive visualization

**Learning Framework:** Semi-supervised active learning with contrastive classification

**Interface:** BokehJS with Python backend for real-time multi-user collaboration

---

## Dataset

Framework evaluated on canine invasive urothelial carcinoma (IncUC) specimens:
- 8 whole slide images at 40× magnification (0.25 μm/pixel)
- Dimensions: 80,000×100,000 pixels per slide
- Annotation categories: mitotic figures, nucleolar characteristics, chromatin texture, nuclear shape

---

## Code Release

**Status:** Code and demo will be available upon paper acceptance.

The repository will include:
- Complete annotation framework implementation
- Preprocessing and segmentation pipelines
- Active learning and embedding generation modules
- Web interface and visualization tools
- Documentation and usage examples

---

## Citation

If you use AnnotateAnyCell in your research, please cite:

```bibtex
@article{annotateanycell2024,
  title={AnnotateAnyCell: Open-Source AI Framework for Efficient Annotation in Digital Pathology},
  author={Verma, Shourya and others},
  journal={},
  year={2024}
}
```

---

## Contact

For questions or collaboration inquiries, please open an issue or contact the authors.

---

## License

This project will be released under the MIT License.
