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
## Dataset Preparation
We utilize a clinically collected dataset. The data should be organized in an unpaired format (similar to CycleGAN). Create a dataset directory with the following structure:
```
src/data
├── dataset_name
│   ├── train_A   # Source domain: White Light Endoscopy (WLE) images
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── train_B   # Target domain: Stained (Chromoendoscopy) images
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── fixed_prompt_a.txt
|   └── fixed_prompt_b.txt
|
|   ├── test_A   # WLE images for testing
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── test_B   # Stained images for testing
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
```
## Inference
We provide an inference script to test your trained model. Since the model utilizes custom weights, you need to specify the --model_path and provide a --prompt (used for the text encoder conditioning) along with the translation --direction.

Translate WLE to Virtual Chromoendoscopy (A → B):
```
python src/inference.py \
    --input_image "path/to/test_A/sample_wle.png" \
    --model_path "outputs/checkpoints/model.pkl" \
    --direction "a2b" \
    --prompt "virtual chromoendoscopy image with indigo carmine" \
    --output_dir "outputs/results"
```
## Training
To train the BCG framework on your own clinically collected dataset, use the train_framework.py script. The training process leverages accelerate for distributed training and mixed precision.
- Initialize the `accelerate` environment with the following command:
    ```
    accelerate config
    ```

- Run the following command to train the model. 
    ```
    export NCCL_P2P_DISABLE=1
    accelerate launch --main_process_port 29501 src/train_framework.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --output_dir="output/BCG/your_dataset" \
        --dataset_folder "data/your_dataset" \
        --train_img_prep "resize_256x256" --val_img_prep "resize_256x256" \
        --learning_rate="1e-5" --max_train_steps=6000 \
        --train_batch_size=1 --gradient_accumulation_steps=1 \
        --report_to "wandb" --tracker_project_name "gparmar_unpaired_vs_cycle_debug_v2" \
        --enable_xformers_memory_efficient_attention --validation_steps 500 \
        --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1
    ```
## Acknowledgment
Our work is built upon the Stable Diffusion-Turbo base model. We thank the authors for their open-source contributions.
