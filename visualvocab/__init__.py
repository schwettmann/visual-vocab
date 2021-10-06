"""Tools for finding, distilling, and annotating LSDs.

This library provides access to the dataset of annotated layer-selective
directions described in "Toward a Visual Concept Vocabulary for GAN Latent
Space" [Schwettmann et al., 2021].

In addition to parsing and loading the BigGAN directions and their annotations,
it also provides generic functions for finding LSDs on arbitary GANs (that have
PyTorch implementations) and distilling those LSDs once they have been
annotated.
"""
