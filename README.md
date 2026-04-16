# Neonatal Jaundice Detection

Detecting neonatal jaundice from demographic images using deep learning — a low-cost, non-invasive screening approach.

## Overview

Jaundice is one of the most common conditions in newborns, affecting up to 60% of full-term babies. Traditional diagnosis requires clinical blood tests. This project explores whether a computer vision model can classify jaundice vs. non-jaundice cases directly from images, enabling faster and more accessible screening.

**Output:** Binary classification — `Jaundice` or `No Jaundice`

---

## Dataset

- **NeoJaundice** — Neonatal Jaundice Evaluation in Demographic Images  
  Published on SpringerNature Figshare: [Dataset Link](https://springernature.figshare.com/articles/dataset/NeoJaundice_Neonatal_Jaundice_Evaluation_in_Demographic_Images/22302559?file=39672982)  
  A real academic dataset containing neonatal images across diverse demographic groups.

- **NJN** — A Dataset for the Normal and Jaundiced Newborns  
  600 newborns, 670 images (560 normal, 200 jaundiced), collected at Al-Elwiya Maternity Teaching Hospital, Baghdad. Includes RGB and YCrCb channel values in CSV format: [Dataset Link](https://sites.google.com/view/neonataljaundice)

---

## Approach

Evaluated and compared multiple pretrained CNN architectures using transfer learning:

- **ResNet**
- **VGG**
- **GoogLeNet**
- **Alexnet**
- **inceptionv3**
- **mobilenet**
- **squeezenet**
- **densenet**

Each model was fine-tuned on the neonatal jaundice dataset. Transfer learning was chosen to leverage features learned from large-scale image datasets and adapt them to this medical imaging task.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training and fine-tuning |
| OpenCV | Image preprocessing |
| NumPy | Data manipulation |
| tqdm | Training progress tracking |

---

## Project Structure

```
├── jaundice_detection.ipynb   # Main training and evaluation notebook
├── NeoJaundice/               # Dataset directory
├── code_snippets/             # Utility scripts
├── requirement.txt            # Dependencies
├── pytroject.toml             # Project declaration and dependencies for uv 
├── extra code                 # similar to main file but with other dataset
```

---

## Installation

```bash
pip install uv
uv venv ./.venv --python 3.11
source ./.venv/bin/activate
uv sync
```

---

## Results

All evaluated models (ResNet, VGG, GoogLeNet) struggled with class imbalance in the dataset — 
models tended to collapse into predicting a single class, resulting in accuracy that simply 
reflected the data split ratio rather than genuine learning. This highlighted the challenge 
of training on imbalanced medical imaging datasets and pointed toward the need for techniques 
like class weighting, oversampling, or data augmentation in future work.


## Notes

This was an exploratory research project comparing pretrained CNN architectures on a real-world medical imaging dataset. The goal was to understand how well off-the-shelf vision models transfer to neonatal jaundice classification.
