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

1. Classical mass and spring mechanical model for phonon vibrations [Tutorial](Tutorial_mass_spring_phonon_model.ipynb)

    A classical mass and spring model can be used to give intuitions for the vibrational phonon modes. With the simple energy functional form (energy stored in the mechanical springs), this tutorial shows the main architecture and usage of the phonax in deriving the second derivative Hessians.

2. Periodic crystalline solids phonon predictions with pre-trained energy models [Tutorial](Tutorial_phonon_with_pretrain_model.ipynb)

    To showcase the phonon band predictions for general crystalline solids, we provide several pre-trained energy models (NqeuIP and MACE, with PBE and PBEsol) that cant be used to derive the phonon properties, given a crystal structure file (vasp format).

3. Train a new energy model for making phonon predictions [Tutorial](Tutorial_new_model_training.ipynb)

    For specific material applications, one can train new energy models that are tailored to these materials by training energy and force data generated for the relevant material types. With a converged energy model, one can make better phonon predictions for these applications. To examine the equilibrium condition for the given crystal structure, one can [check](Tutorial_model_force_check.ipynb) the atomic forces, which should be vanishing for a meaningful phonon prediction.

4. IR / Raman optical spectroscopy activities for vibrational states [Tutorial](Tutorial_CH4_molecule_IR_Raman_symmetry.ipynb)

    Besides making vibrational mode energy predictions, phonax can also be used to analyze the vibrational state symmetry properties and their activities under IR / Raman optical spectroscopy probes from the symmetry selection rules.

5. Hessian training for molecules [Tutorial](Tutorial_molecular_hessian_training.ipynb)

   In this notebook, we demonstrate how to improve and fine-tune the energy model by augmented training with molecular Hessians or the vibrational spectrum (eigenvalues of the dynamical matrix from hessians)

6. Hessian training for crystals [Tutorial](Tutorial_crystal_hessian_training.ipynb)

   Similarily, the Hessian training can also be used for the periodic crystals to improve the energy model and foce predictions.   

7. Gradio phonax prediction interface

   To launch this gradio web interface for using phonax, run the follwing command in the download folder:
   ```bash
   python launch_gradio_service.py
   ``` 
   A default local URL will be generated at http://0.0.0.0:7860 which can be opened in any web browser.
   In this gradio web interface, one can simply upload a structural file in cif format, and choose the prediction setting (select the fine-tuned model for PBEsol predictions, otherwise PBE based predictions).
   The phonon band structure and the (projected) phonon density of state will be computed.

## References

1. [e3nn: Euclidean Neural Networks](https://arxiv.org/abs/2207.09453)

2. [NequIP: E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials](https://www.nature.com/articles/s41467-022-29939-5)

3. [MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2206.07697), [The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials](https://arxiv.org/abs/2205.06643)

4. [JAX M.D. A Framework for Differentiable Physics](https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf)

5. [PoSym: A python library to analyze the symmetry of theoretical chemistry objects](https://zenodo.org/records/7261326)

## Citation

1. [NeurIPS 2023 AI4Mat workshop](https://openreview.net/forum?id=xxyHjer00Y)


