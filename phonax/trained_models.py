import json
import os
import pickle
import random
import sys
import pickle

from typing import Callable, Dict, List, Optional
import haiku as hk
import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml


from phonax.datasets import datasets
from phonax.utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
)
from phonax.data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)
from phonax.predictors import predict_energy_forces_stress
from phonax.optimizer import optimizer
from phonax.energy_force_train import energy_force_train
from phonax.loss import WeightedEnergyFrocesStressLoss


from phonax.nequip_model import NequIP_JAXMD_model, NequIP_JAX_model
from phonax.mace_model import MACE_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
np.set_printoptions(precision=3, suppress=True)


def NequIP_JAXMD_uniIAP_model(weights_path):
    reload = os.path.join(weights_path, "nequip-uniIAP")

    with open(os.path.join(weights_path, "nequip-uniIAP", "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    save_dir_name = None
    
    model_fn, params, num_message_passing = NequIP_JAXMD_model(
        r_max=config['cutoff'],
        atomic_energies_dict={},
        train_graphs=[],
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        use_sc = True,
        graph_net_steps = config["model"]["num_layers"],
        hidden_irreps = config["model"]["internal_irreps"],
        nonlinearities =  {'e': 'swish', 'o': 'tanh'},
        save_dir_name = save_dir_name,
        reload = reload,
    )
    
    return model_fn, params, num_message_passing, config['cutoff']
    


def NequIP_JAXMD_molecule_model(weights_path):
    reload = os.path.join(weights_path, "nequip-molecule")

    with open(os.path.join(weights_path, "nequip-molecule", "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_dir_name = None

    model_fn, params, num_message_passing = NequIP_JAXMD_model(
        r_max=config['cutoff'],
        atomic_energies_dict={},
        train_graphs=[],
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        use_sc = True,
        graph_net_steps = config["model"]["num_layers"],
        hidden_irreps = config["model"]["internal_irreps"],
        nonlinearities =  {'e': 'swish', 'o': 'tanh'},
        save_dir_name = save_dir_name,
        reload = reload,
    )

    return model_fn, params, num_message_passing, config['cutoff']


    
def NequIP_JAXMD_uniIAP_PBEsol_finetuned_model(weights_path):
    reload = os.path.join(weights_path, "nequip-uniIAP-finetune")

    with open(os.path.join(weights_path, "nequip-uniIAP-finetune", "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    save_dir_name = None
    
    model_fn, params, num_message_passing = NequIP_JAXMD_model(
        r_max=config['cutoff'],
        atomic_energies_dict={},
        train_graphs=[],
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        use_sc = True,
        graph_net_steps = config["model"]["num_layers"],
        hidden_irreps = config["model"]["internal_irreps"],
        nonlinearities =  {'e': 'swish', 'o': 'tanh'},
        save_dir_name = save_dir_name,
        reload = reload,
    )
    
    return model_fn, params, num_message_passing, config['cutoff']


def MACE_uniIAP_model(weights_path):
    reload = os.path.join(weights_path, "mace-uniIAP")

    with open(os.path.join(weights_path, "mace-uniIAP", "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    save_dir_name = None
    
    model_fn, params, num_message_passing = MACE_model(
        r_max=config['cutoff'],
        atomic_energies_dict={},
        train_graphs=[],
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        hidden_irreps = "128x0e + 128x1o",
        readout_mlp_irreps = "32x0e",
        max_ell = 3,
        num_interactions = 2,
        interaction_irreps = "o3_restricted",
        epsilon = 0.4,
        correlation = 3,
        path_normalization = "path",
        gradient_normalization = "path",
        save_dir_name = save_dir_name,
        reload = reload,
    )
    
    return model_fn, params, num_message_passing, config['cutoff']
    
    
def MACE_uniIAP_PBEsol_finetuned_model(weights_path):
    reload = os.path.join(weights_path, "mace-uniIAP-finetune")

    with open(os.path.join(weights_path, "mace-uniIAP-finetune", "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    save_dir_name = None
    
    model_fn, params, num_message_passing = MACE_model(
        r_max=config['cutoff'],
        atomic_energies_dict={},
        train_graphs=[],
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        hidden_irreps = "128x0e + 128x1o",
        readout_mlp_irreps = "32x0e",
        max_ell = 3,
        num_interactions = 2,
        interaction_irreps = "o3_restricted",
        epsilon = 0.4,
        correlation = 3,
        path_normalization = "path",
        gradient_normalization = "path",
        save_dir_name = save_dir_name,
        reload = reload,
    )
    
    return model_fn, params, num_message_passing, config['cutoff']
    
    

def NequIP_JAX_uniIAP_model(weights_path):
    reload = os.path.join(weights_path, "nequip-JAX-uniIAP")

    with open(os.path.join(weights_path, "nequip-JAX-uniIAP", "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    save_dir_name = None
    
    model_fn, params, num_message_passing = NequIP_JAX_model(
        r_max=config['cutoff'],
        atomic_energies_dict={},
        avg_num_neighbors = config["model"]["normalization_constant"],
        train_graphs=[],
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        use_sc = True,
        graph_net_steps = config["model"]["num_layers"],
        hidden_irreps = config["model"]["internal_irreps"],
        nonlinearities =  {'e': 'swish', 'o': 'tanh'},
        save_dir_name = save_dir_name,
        reload = reload,
    )
    
    return model_fn, params, num_message_passing, config['cutoff']
    
    

