# Toward a Visual Concept Vocabulary for GAN Latent Space <br><sub>Code and data from the ICCV 2021 paper</sub>


[**Paper**](https://openaccess.thecvf.com/content/ICCV2021/html/Schwettmann_Toward_a_Visual_Concept_Vocabulary_for_GAN_Latent_Space_ICCV_2021_paper.html)  |
[**Website**]( https://visualvocab.csail.mit.edu/) |
[**arxiv**](https://arxiv.org/pdf/2110.04292.pdf) | Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schwettmann/visual-vocab/blob/main/visualvocab/notebooks/demo.ipynb)<br>
[Sarah Schwettmann](https://cogconfluence.com), [Evan Hernandez](https://evandez.com/), [David Bau](http://davidbau.com/), [Samuel Klein](http://blogs.harvard.edu/sj/), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <br>
MIT CSAIL, MIT BCS

This repository contains code for loading the vocabulary of visual concepts in BigGAN used in the original paper and reproducing our results. Additionally we provide code for generating new layer-selective directions, and disentangling them into a vocabulary of visual concepts using your own corpus of annotations. 

## Overview

![teaser_final_cmu-01](https://user-images.githubusercontent.com/26309530/137186304-0c89f9bc-3f74-4b93-8972-245605cad2a7.png)

## Installation

The provided code has been tested for Python 3.8 on MacOS and Ubuntu 20.04. 

To run the code yourself, start by cloning the repository:
```bash
git clone https://github.com/schwettmann/visual-vocab
cd visual-vocab
```
(**Optional**) You will probably want to create a conda environment or virtual environment instead of installing the dependencies globally. E.g., to create a new virtual environment you can run:
```bash
python3 -m venv env
source env/bin/activate
```
Finally, install the Python dependencies using pip:
```bash
pip3 install -r requirements.txt
```

## Usage

To download any of the various annotated directions from the paper, use `datasets.load` submodule. It downloads and parses the annoated directions. Example usage:
```python
from visualvocab import datasets

# Download layer-selective directions and annotations used for distilling single-word directions:
dataset = datasets.load('lsd_all')

# Download distilled directions for all BigGAN-Places365 categories:
dataset = datasets.load('distilled_all')

# Download distilled directions for a specific BigGAN-Places365 category:
dataset = datasets.load('distilled_cottage')
```
See the module for a full list of available annotated directions.

## Citation

If you use this code for your research, please cite [our paper](https://arxiv.org/pdf/2110.04292.pdf) : 

```bibtex
@InProceedings{Schwettmann_2021_ICCV,
    author    = {Schwettmann, Sarah and Hernandez, Evan and Bau, David and Klein, Samuel and Andreas, Jacob and Torralba, Antonio},
    title     = {Toward a Visual Concept Vocabulary for GAN Latent Space},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6804-6812}
}
```
