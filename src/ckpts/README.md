
# Model Checkpoints

Model weight files are **not uploaded to this GitHub repository** due to file size limits and double-blind peer review policies. Please download the pretrained weights from our anonymous Hugging Face repository and place them in the designated folder.

## Download Link

You can access and download the model checkpoint directly from Hugging Face:

* **Hugging Face Repository**: [BCG-anonymous/BCG](https://huggingface.co/BCG-anonymous/BCG)
* **Direct Download**: [Download model.pkl](https://huggingface.co/BCG-anonymous/BCG/resolve/main/model.pkl)

Alternatively, you can download the weight file directly via terminal using `wget`:
```bash
wget [https://huggingface.co/BCG-anonymous/BCG/resolve/main/model.pkl](https://huggingface.co/BCG-anonymous/BCG/resolve/main/model.pkl) -P src/ckpts/
```

## Directory Structure
After downloading, ensure your file structure follows:
```
src/ckpts/
└──your_model.pkl
```
