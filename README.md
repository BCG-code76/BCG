# BCG: Bidirectional Color-Guided Diffusion Framework for Virtual Chromoendoscopy

 [**Code**](https://github.com/BCG-code76/BCG) 

Virtual staining of endoscopic images offers a safer, AI-driven alternative to physical chromoendoscopy. However, accurately rendering precise diagnostic dye textures while preserving underlying anatomical structures remains a fundamental clinical challenge. 

We propose a novel **Bidirectional Color-Guided (BCG)** framework for unpaired virtual chromoendoscopy. By integrating a **Prior Condition Encoder (PCE)** for explicit spectral injection and an **HSI-aware Histogram Mutual Information (HMI) loss** for statistical alignment, our method bypasses the spatial compression bottleneck of standard diffusion models. It achieves state-of-the-art spectral fidelity and structural preservation with highly efficient one-step inference.

<br>
<div>
<p align="center">
<img src='https://raw.githubusercontent.com/BCG-code76/BCG/main/src/assets/result.jpg' align="center" width=900px>
</p>
</div>

---

## Method
**Our Framework Architecture:**
To decouple spectral preservation from spatial compression, we introduce a Prior Condition Encoder (PCE) that aggregates spectral information from explicit density, statistical moments, and implicit semantics. This information is injected into the denoising backbone via cross-attention. Furthermore, we enforce spectral alignment through a hue-saturation-intensity (HSI)-aware HMI loss, maximizing the statistical dependency between generated and target domains.

<div>
<p align="center">
<img src='https://raw.githubusercontent.com/BCG-code76/BCG/main/src/assets/method.jpg' align="center" width=900px>
</p>
</div>

---

## Getting Started

### Environment Setup
We recommend using Conda to set up the environment, inheriting the dependencies from the SD-Turbo base model.
```
bash
conda env create -f environment.yaml
conda activate BCG
```
Or use a virtual environment:
```
python3 -m venv BCG
source venv/bin/activate
pip install -r requirements.txt
```
