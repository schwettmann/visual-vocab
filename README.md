# Toward a Visual Concept Vocabulary for GAN Latent Space
**Code and data from the ICCV 2021 paper**
![teaser_final_cmu-01](https://user-images.githubusercontent.com/26309530/137186304-0c89f9bc-3f74-4b93-8972-245605cad2a7.png)

**Notice:** This repository is under active development! Expect instability until at least October 25th, 2021.

This repository contains code for finding layer-selective directions, distilling them, and loading all of the BigGAN directions used in the original paper.

## Installation

The provided code has been tested for Python 3.8 on MacOS and Ubuntu 20.04. It may still work in other environments, but we make no guarantees.

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

*Notice: This section is under construction and will be updated as functionality gets added.*

To download any of the various annotated directions from the paper, use `datasets.load` submodule. It downloads and parses the annoated directions. Example usage:
```python
from visualvocab import datasets

# Download layer-selective directions and annotations used for distilling single-word directions:
dataset = datasets.load('lsd_2x')

# Download distilled directions for a specific BigGAN-Places365 category:
dataset = datasets.load('distilled_cottage')

# Download distilled directions for all BigGAN-Places365 categories:
dataset = datasets.load('distilled_all')
```
See the module for a full list of available annotated directions.

## Citation

Sarah Schwettmann, Evan Hernandez, David Bau, Samuel Klein, Jacob Andreas, Antonio Torralba. *Toward a Visual Concept Vocabulary for GAN Latent Space*, Proceedings of the International Conference on Computer Vision (ICCV), 2021.

## Bibtex

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
