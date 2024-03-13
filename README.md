# Phonax

Library to train jax models with phonon data.


## Quick Start

The following steps will guid you through the installation for the phonax library and the tests for tutorial notebooks.

1. In a conda env with JAX / CUDA available clone and download this phonax repository.

2. Download the pre-trained model weights and the example datasets.
    ```bash
    bash xxxx download_script
    ```

3. In the downloaded directory run:
    ```bash
    pip install -e .
    ```
4. Install the additional libraries from github repositories:
    ```bash
    pip install -r requirements_extra.txt
    ```




## Tutorials and scripts

### Mass and spring mechanical model for phonon vibrations


### Periodic crystalline solids


### Molecules

### Gradio phonax prediction interface


To launch this gradio web interface for using phonax, run the follwing command in the download folder:

    ```bash
    python launch_gradio_service.py
    ``` 

A local URL will be generated at http://0.0.0.0:7860 which can be opened in any web browser.
In this gradio web interface, one can simply upload a structural file in cif format, and choose the prediction setting (select the fine-tuned model for PBEsol predictions, otherwise PBE based predictions).
The phonon band structure and the (projected) phonon density of state will be computed.

## References

1. [NeurIPS 2023 AI4Mat workshop](https://openreview.net/forum?id=xxyHjer00Y)

2. [arXiv](https://arxiv.org/)

## Citation


