# Phonax

Phonax is a JAX framework to use E(3)-equivariant neural network for predicting vibrational and phonon modes of molecules and periodic crystals.
Alternatively, Phonax also enables the utilization of the second-order derivative Hessians or the vibrational spectrum as eigenvalues to fine-tune the underlying energy models.


## Quick Start

The following steps will guide you through the installation for the phonax library and the tests with tutorial notebooks.

1. In a conda environment with JAX / CUDA available, clone and download this phonax repository.

2. In the downloaded directory run:
    ```bash
    pip install -e .
    ```
3. Install the additional libraries from github repositories:
    ```bash
    pip install -r requirements_extra.txt
    ```
4. [Download](https://figshare.com/s/66bf63d6b5ff42638184) the pre-trained model weights and the datasets, and uncompress the file at the folder with the tutorial notebooks.

    ```bash
    tar zxvf phonax-download.tar.gz
    ```

## Tutorials and scripts

### Classical mass and spring mechanical model for phonon vibrations [Tutorial](Tutorial_mass_spring_phonon_model.ipynb)
A classical mass and spring model can be used to give intuitions for the vibrational phonon modes. With the simple energy functional form (energy stored in the mechanical springs), this tutorial shows the main architecture and usage of the phonax in deriving the second derivative Hessians.


### Periodic crystalline solids


### Molecules

### Gradio phonax prediction interface


To launch this gradio web interface for using phonax, run the follwing command in the download folder:
```bash
python launch_gradio_service.py
``` 

A default local URL will be generated at http://0.0.0.0:7860 which can be opened in any web browser.
In this gradio web interface, one can simply upload a structural file in cif format, and choose the prediction setting (select the fine-tuned model for PBEsol predictions, otherwise PBE based predictions).
The phonon band structure and the (projected) phonon density of state will be computed.

## References

1. [NeurIPS 2023 AI4Mat workshop](https://openreview.net/forum?id=xxyHjer00Y)

2. [arXiv](https://arxiv.org/)

## Citation


