import logging
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm

import ase
import ase.io


from phonax.phonons import (
    DataLoader,
    atoms_to_ext_graph,
)

from .data_utils import (
    load_from_xyz,
    random_train_valid_split,
    graph_from_configuration,
    GraphDataLoader,
    load_config,
)


def datasets(
    *,
    r_max: float,
    config_dataset: dict,
) -> Tuple[
    GraphDataLoader,
    GraphDataLoader,
    GraphDataLoader,
    Dict[int, float],
    float,
]:
    """Load training and test dataset from xyz file"""
    
    train_path = load_config(config_dataset,'train_path',None)
    valid_path = load_config(config_dataset,'valid_path',None)
    test_path = load_config(config_dataset,'test_path',None)
    train_num = load_config(config_dataset,'train_num',None)
    valid_num = load_config(config_dataset,'valid_num',None)
    valid_fraction = load_config(config_dataset,'valid_fraction',None)
    test_num = load_config(config_dataset,'test_num',None)
    
    config_type_weights = load_config(config_dataset,'config_type_weights')
    seed = load_config(config_dataset,'seed',1234)
    loader_seed= load_config(config_dataset,'loader_seed',5678)
    energy_key= load_config(config_dataset,'energy_key',"energy")
    forces_key= load_config(config_dataset,'forces_key',"forces")
    n_node = load_config(config_dataset,'n_node',1)
    n_edge = load_config(config_dataset,'n_edge',1)
    n_graph = load_config(config_dataset,'n_graph',1)
    min_n_node = load_config(config_dataset,'min_n_node',1)
    min_n_edge = load_config(config_dataset,'min_n_edge',1)
    min_n_graph = load_config(config_dataset,'min_n_graph',1)
    n_mantissa_bits = load_config(config_dataset,'n_mantissa_bits',1)
    prefactor_stress = load_config(config_dataset,'prefactor_stress',1.0)
    remap_stress= load_config(config_dataset, 'remap_stress',None)

    print('nums check',n_node,n_edge,n_graph)

    atomic_energies_dict_xyz, all_train_configs = load_from_xyz(
        file_or_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        extract_atomic_energies=False,
        num_configs=train_num,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
    )

    print(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )

    if valid_path is not None:
        _, valid_configs = load_from_xyz(
            file_or_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            num_configs=valid_num,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        print(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    elif valid_fraction is not None:
        print(
            f"Using random {100 * valid_fraction}% of training set for validation"
        )
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, int(len(all_train_configs) * valid_fraction), seed
        )
    elif valid_num is not None:
        print(f"Using random {valid_num} configurations for validation")
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, valid_num, seed
        )
    else:
        print("No validation set")
        train_configs = all_train_configs
        valid_configs = []
    del all_train_configs

    if test_path is not None:
        _, test_configs = load_from_xyz(
            file_or_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            num_configs=test_num,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        print(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    else:
        test_configs = []

    print(
        f"Total number of configurations: "
        f"train={len(train_configs)}, "
        f"valid={len(valid_configs)}, "
        f"test={len(test_configs)}"
    )

    train_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration(c, cutoff=r_max) for c in tqdm(train_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=True,
        return_remainder=False,
        loader_seed = loader_seed,
    )
    valid_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration(c, cutoff=r_max) for c in tqdm(valid_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        return_remainder=True,
        #loader_seed = loader_seed,
    )
    test_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration(c, cutoff=r_max) for c in tqdm(test_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        return_remainder=True,
        shuffle=False,
        # loader_seed = loader_seed,
    )
    return train_loader, valid_loader, test_loader,  r_max # atomic_energies_dict_xyz


def datasets_groups(
    valid_paths,
    r_max: float,
    config_type_weights: Dict = None,
    train_num: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    valid_num: int = None,
    test_path: str = None,
    test_num: int = None,
    seed: int = 1234,
    loader_seed: int = 5678,
    energy_key: str = "energy",
    forces_key: str = "forces",
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
    min_n_node: int = 1,
    min_n_edge: int = 1,
    min_n_graph: int = 1,
    n_mantissa_bits: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
) -> List:
    """Load training and test dataset from xyz file"""
    
    valid_loaders = []
    
    for valid_path in valid_paths:
        valid_name = valid_path.split('/')[-1]   #.split('.')[0]
        _, valid_configs = data.load_from_xyz(
            file_or_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            #num_configs=valid_num,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        #print('vconfigs',valid_configs)

        valid_loader = GraphDataLoader(
            graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm.tqdm(valid_configs)
            ],
            n_node=n_node,
            n_edge=n_edge,
            n_graph=n_graph,
            min_n_node=min_n_node,
            min_n_edge=min_n_edge,
            min_n_graph=min_n_graph,
            n_mantissa_bits=n_mantissa_bits,
            shuffle=False,
            #loader_seed = loader_seed,
        )
        
        valid_loaders.append((valid_name,valid_loader))
   
    return valid_loaders



def ph_datasets(
    *,
    r_max: float,
    config_dataset: dict,
    num_message_passing: int,
    #seed: int = 0,
):
    """Load training and test dataset from xyz file"""
    train_path = load_config(config_dataset,'train_path',None)
    valid_path = load_config(config_dataset,'valid_path',None)
    test_path = load_config(config_dataset,'test_path',None)
    train_num = load_config(config_dataset,'train_num',None)
    valid_num = load_config(config_dataset,'valid_num',None)
    valid_fraction = load_config(config_dataset,'valid_fraction',None)
    test_num = load_config(config_dataset,'test_num',None)
    
    #config_type_weights = load_config(config_dataset,'config_type_weights')
    seed = load_config(config_dataset,'seed',1234)
    #loader_seed= load_config(config_dataset,'loader_seed',5678)
    #energy_key= load_config(config_dataset,'energy_key',"energy")
    #forces_key= load_config(config_dataset,'forces_key',"forces")
    n_node = load_config(config_dataset,'n_node',1)
    n_edge = load_config(config_dataset,'n_edge',1)
    n_graph = load_config(config_dataset,'n_graph',1)
    #min_n_node = load_config(config_dataset,'min_n_node',1)
    #min_n_edge = load_config(config_dataset,'min_n_edge',1)
    #min_n_graph = load_config(config_dataset,'min_n_graph',1)
    #n_mantissa_bits = load_config(config_dataset,'n_mantissa_bits',1)

    def load_graphs(path, name, num=None):
        #### paths = sorted(glob(path))??
        paths = [path]

        all_atoms = []
        for path in paths:
            all_atoms += ase.io.read(path, format="extxyz", index=":")

        if num is not None:
            all_atoms = all_atoms[:num]

        all_graphs = [
            atoms_to_ext_graph(atoms, r_max, num_message_passing)
            for atoms in tqdm(all_atoms, desc=f"Pad {name}")
        ]
        original_length = len(all_graphs)

        all_graphs = [
            graph
            for graph in all_graphs
            if (graph.n_node[0] <= n_node - 1 and graph.n_edge[0] <= n_edge)
        ]

        if len(all_graphs) != original_length:
            logging.info(
                f"Removed {original_length - len(all_graphs)} graphs from {name} set "
                f"with n_node > {n_node - 1} or n_edge > {n_edge}"
            )
        return all_graphs

    all_train_graphs = load_graphs(train_path, "train", num=train_num)

    if valid_path is not None:
        train_graphs, valid_graphs = all_train_graphs, load_graphs(
            valid_path, "valid", num=valid_num
        )
    else:
        train_graphs, valid_graphs = data.random_train_valid_split(
            all_train_graphs, valid_fraction, seed
        )

    test_graphs = (
        load_graphs(test_path, "test", num=test_num) if test_path is not None else []
    )

    print(
        f"Total number of configurations: "
        f"train={len(train_graphs)}, "
        f"valid={len(valid_graphs)}, "
        f"test={len(test_graphs)}"
    )

    train_loader = DataLoader(
        train_graphs,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_graphs,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_graphs,
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader, r_max, num_message_passing

