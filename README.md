# Toward a Visual Concept Vocabulary for GAN Latent Space <br><sub>Code and data from the ICCV 2021 paper</sub>


[**Paper**](https://openaccess.thecvf.com/content/ICCV2021/html/Schwettmann_Toward_a_Visual_Concept_Vocabulary_for_GAN_Latent_Space_ICCV_2021_paper.html)  |
[**Website**]( https://visualvocab.csail.mit.edu/) |
[**arxiv**](https://arxiv.org/pdf/2110.04292.pdf) | Tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schwettmann/visual-vocab/blob/main/visualvocab/notebooks/vcv_demo.ipynb)<br>
[Sarah Schwettmann](https://cogconfluence.com), [Evan Hernandez](https://evandez.com/), [David Bau](http://davidbau.com/), [Samuel Klein](http://blogs.harvard.edu/sj/), [Jacob Andreas](https://www.mit.edu/~jda/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/) <br>
MIT CSAIL, MIT BCS

This repository contains code for loading and visualizing the vocabulary of visual concepts in BigGAN used in the original paper and reproducing our results. Additionally we provide code for generating new layer-selective directions that can be disentangled into a vocabulary of visual concepts using your own corpus of annotations.

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

To download the vocabulary from the paper, use the `datasets.load` submodule. It downloads and parses the annoated directions. Example usage:
```python
from visualvocab import datasets

# Download layer-selective directions and annotations used for distilling single-word directions:
dataset = datasets.load('lsd_all')

# Download distilled directions for all BigGAN-Places365 categories:
dataset = datasets.load('distilled_all')

# Download distilled directions for a specific BigGAN-Places365 category:
dataset = datasets.load('distilled_cottage')
```
See the module for a full list of available annotated directions. You can experiment with loading and visualizing our precomputed vocabulary in the demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schwettmann/visual-vocab/blob/main/visualvocab/notebooks/vcv_demo.ipynb)

### Generating your own layer-selective directions
In addition to downloading our visual concepts and associated directions, you may also want to generate your own. Why? There are multiple reasons to want to constuct your own visual concept vocabulary:
  1. To capture concepts shared by a model's representation space and _your own perception_, or that of a particular group of observers (e.g. experts in some area).
  2. To compare concepts represented by different models, or models trained on different datasets.
 
Our method for extracting a set of disentangled, human-recognizable concepts works with any corpus of annotated directions; however, we achived best results (a more diverse vocabulary with higher agreement between annotators) when using the procedure we describe for obtaining an initial set of _layer-selective directions_ (LSDs) for annotation. We provide code for generating new LSDs in the `generate_lsds` module. 

Example usage:
```python
from visualvocab import generate_lsds 

# Training parameters 
batch_size = 1                     # for generator, do not change this
new_class  = 203                   # Places 365 image class (change)
learning_rate = 0.01               # you can also try changing this as well
num_dirs_per_layer = 4             # number of LSDs you want to generate per layer
num_samples = 2000                 # num samples for optimization. Training on 2000 samples should get you a desired result.
start_layer = 2                    # layer where you want to start generating LSDs (higher layer numbers are closer to image output)
end_layer = 0                      # last layer for which you want to find LSDs. leave end_layer = 0 and you will calcualate LSDs for all layers before (&incl.) start layer. 
visualize = True                   # do you want to visualize each direction after it is generated? 
savedirs = False                   # do you want to save the directions? 
savepath = '/mydirectory/LSDs/'    # set to your own path where you want to save the directions

z, _ = utils.prepare_z_y(batch_size, G.dim_z, n_classes, device=device, z_var=0.5)   #generate a random z to start 

directions = generate_lsds.optimize_lsds(z, num_dirs_per_layer, num_samples, start_layer, end_layer, visualize, learning_rate, new_class, savedirs, savepath)

```
Requirements: `generate_lsds` loads a pretrained BigGAN and finds directions inside its latent space. By default this code runs on BigGAN-Places. It is easy to modify it to instead run on BigGAN-Imagenet (change `pretrained = 'places365'` to `pretrained = 'places365'`). We also encourage you to try this method with your own pretrained models, but that will require more customization. 

## Example Results

![example concepts from the paper](https://github.com/schwettmann/visual-vocab/blob/main/visualvocab/example_concepts.jpg?raw=true)

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
