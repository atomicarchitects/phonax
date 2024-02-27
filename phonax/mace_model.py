import functools
import logging
import math
from typing import Callable, Dict, List, Optional, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pickle
import json


#from ..tools import get_edge_relative_vectors, safe_norm, sum_nodes_of_the_same_graph


from .blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from phonax.data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)
from phonax.utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
    bessel_basis,
    soft_envelope,
    safe_norm,
)



try:
    from profile_nn_jax import profile
except ImportError:

    def profile(_, x):
        return x


class GeneralMACE(hk.Module):
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        r_max: float,
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        avg_r_min: float = None,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        # Number of zero derivatives at small and large distances, default 4 and 2
        # If both are None, it uses a smooth C^inf envelope function
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        gate: Callable = jax.nn.silu,  # activation function
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        node_embedding: hk.Module = LinearNodeEmbeddingBlock,
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)[1:]  # discard 0e

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.r_max = r_max
        self.correlation = correlation
        self.avg_num_neighbors = avg_num_neighbors
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

        # Embeddings
        self.node_embedding = node_embedding(
            self.num_species, self.num_features * self.hidden_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
        )

    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_specie.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        # Embeddings
        node_feats = self.node_embedding(node_specie).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        node_feats = profile("embedding: node_feats", node_feats)

        lengths = safe_norm(vectors, axis=-1)

        edge_attrs = e3nn.concatenate(
            [
                self.radial_embedding(lengths),
                e3nn.spherical_harmonics(
                    self.sh_irreps,
                    vectors / lengths[..., None],
                    normalize=False,
                    normalization="component",
                ),
            ]
        )  # [n_edges, irreps]

        edge_attrs = profile("embedding: edge_attrs", edge_attrs)

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.activation,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                name=f"layer_{i}",
            )(
                node_feats,
                node_specie,
                edge_attrs,
                senders,
                receivers,
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        return e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


class MACELayer(hk.Module):
    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        num_features: int,
        interaction_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        activation: Callable,
        num_species: int,
        epsilon: Optional[float],
        name: Optional[str],
        # InteractionBlock:
        avg_num_neighbors: float,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
    ) -> None:
        super().__init__(name=name)

        self.first = first
        self.last = last
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.activation = activation
        self.num_species = num_species
        self.epsilon = epsilon
        self.correlation = correlation
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ):
        node_feats = profile(f"{self.name}: node_feats", node_feats)

        sc = None
        if not self.first:
            sc = e3nn.haiku.Linear(
                self.num_features * self.hidden_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp",
            )(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]
            sc = profile(f"{self.name}: self-connexion", sc)

        node_feats = InteractionBlock(
            target_irreps=self.num_features * self.interaction_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            activation=self.activation,
        )(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            receivers=receivers,
            senders=senders,
        )

        if self.epsilon is not None:
            node_feats *= self.epsilon
        else:
            node_feats /= jnp.sqrt(self.avg_num_neighbors)

        node_feats = profile(f"{self.name}: node_feats after interaction", node_feats)

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.haiku.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)
            node_feats = profile(
                f"{self.name}: node_feats after skip_tp_first", node_feats
            )
            sc = None

        node_feats = EquivariantProductBasisBlock(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)

        node_feats = profile(f"{self.name}: node_feats after tensor power", node_feats)

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutBlock(self.output_irreps)(
                node_feats
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutBlock(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
            )(
                node_feats
            )  # [n_nodes, output_irreps]

        node_outputs = profile(f"{self.name}: node_outputs", node_outputs)
        return node_outputs, node_feats


# TODO REMOVE and use predictor instead
class MACE(hk.Module):
    def __init__(
        self,
        *,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
        atomic_energies: np.ndarray,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )
        self.energy_model = GeneralMACE(output_irreps="0e", **kwargs)
        self.atomic_energies = jnp.asarray(atomic_energies)

    def __call__(self, graph: jraph.GraphsTuple) -> Dict[str, jnp.ndarray]:
        def energy_fn(positions):
            vectors = get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=graph.globals.cell,
                n_edge=graph.n_edge,
            )

            contributions = self.energy_model(
                vectors, graph.nodes.species, graph.senders, graph.receivers
            )  # [n_nodes, num_interactions, 0e]

            contributions = contributions.array[:, :, 0]  # [n_nodes, num_interactions]
            node_energies = self.scale_shift(
                jnp.sum(contributions, axis=1)
            )  # [n_nodes, ]

            node_energies += self.atomic_energies[graph.nodes.species]  # [n_nodes, ]
            return jnp.sum(node_energies), node_energies

        minus_forces, node_energies = jax.grad(energy_fn, has_aux=True)(
            graph.nodes.positions
        )

        graph_energies = sum_nodes_of_the_same_graph(
            graph, node_energies
        )  # [ n_graphs,]

        return {
            "energy": graph_energies,  # [n_graphs,]
            "forces": -minus_forces,  # [n_nodes, 3]
        }

    
    
def MACE_model(
    *,
    r_max: float,
    atomic_energies_dict: Dict[int, float] = None,
    train_graphs: List[jraph.GraphsTuple] = None,
    initialize_seed: Optional[int] = None,
    scaling: Callable = None,
    atomic_energies: Union[str, np.ndarray, Dict[int, float]] = None,
    avg_num_neighbors: float = "average",
    avg_r_min: float = None,
    num_species: int = None,
    num_interactions=3,
    path_normalization="path",
    gradient_normalization="path",
    learnable_atomic_energies=False,
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    save_dir_name = None,
    reload = None,
    **kwargs,
):
    if reload is None:
        mace_model_setup = {}
        
        if train_graphs is None:
            z_table = None
        else:
            z_table = get_atomic_number_table_from_zs(
                z for graph in train_graphs for z in graph.nodes.species
            )
        print(f"z_table= {z_table}")
        
        mace_model_setup['z_table'] = z_table

        if avg_num_neighbors == "average":
            avg_num_neighbors = compute_avg_num_neighbors(train_graphs)
            print(
                f"Compute the average number of neighbors: {avg_num_neighbors:.3f}"
            )
        else:
            print(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")

        mace_model_setup['avg_num_neighbors'] = avg_num_neighbors
            
        if avg_r_min == "average":
            avg_r_min = compute_avg_min_neighbor_distance(train_graphs)
            print(f"Compute the average min neighbor distance: {avg_r_min:.3f}")
        elif avg_r_min is None:
            print("Do not normalize the radial basis (avg_r_min=None)")
        else:
            print(f"Use the average min neighbor distance: {avg_r_min:.3f}")

        mace_model_setup['avg_r_min'] = avg_r_min
            
        if atomic_energies is None:
            if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
                atomic_energies = "average"
            else:
                atomic_energies = "isolated_atom"

        if atomic_energies == "average":
            atomic_energies_dict = compute_average_E0s(train_graphs, z_table)
            print(
                f"Computed average Atomic Energies using least squares: {atomic_energies_dict}"
            )
            atomic_energies = np.array(
                [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
            )
        elif atomic_energies == "isolated_atom":
            print(
                f"Using atomic energies from isolated atoms in the dataset: {atomic_energies_dict}"
            )
            atomic_energies = np.array(
                [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
            )
        elif atomic_energies == "zero":
            print("Not using atomic energies")
            atomic_energies = np.zeros(num_species)
        elif isinstance(atomic_energies, np.ndarray):
            print(
                f"Use Atomic Energies that are provided: {atomic_energies.tolist()}"
            )
            if atomic_energies.shape != (num_species,):
                print(
                    f"atomic_energies.shape={atomic_energies.shape} != (num_species={num_species},)"
                )
                raise ValueError
        elif isinstance(atomic_energies, dict):
            atomic_energies_dict = atomic_energies
            print(f"Use Atomic Energies that are provided: {atomic_energies_dict}")
            atomic_energies = np.array(
                [atomic_energies_dict.get(z, 0.0) for z in range(num_species)]
            )
        else:
            raise ValueError(f"atomic_energies={atomic_energies} is not supported")

        mace_model_setup['atomic_energies'] = atomic_energies
            
        if scaling is None:
            mean, std = 0.0, 1.0
        else:
            mean, std = scaling(train_graphs, atomic_energies)
            print(
                f"Scaling with {scaling.__qualname__}: mean={mean:.2f}, std={std:.2f}"
            )
        mace_model_setup['mean'] = mean
        mace_model_setup['std'] = std
            
        if save_dir_name:
            with open(f"{save_dir_name}/mace_model_setup.pkl", "wb") as f:
                pickle.dump(mace_model_setup, f) 

    else:
        with open(f"{reload}/mace_model_setup.pkl", "rb") as f:
            mace_model_setup = pickle.load(f)
            
        if save_dir_name:
            with open(f"{save_dir_name}/mace_model_setup.pkl", "wb") as f:
                pickle.dump(mace_model_setup, f) #  E0s.tolist()
        
        z_table = mace_model_setup['z_table']
        avg_num_neighbors = mace_model_setup['avg_num_neighbors']
        avg_r_min = mace_model_setup['avg_r_min']
        atomic_energies = mace_model_setup['atomic_energies']
        mean = mace_model_setup['mean']
        std = mace_model_setup['std']
        
        
        
    # check that num_species is consistent with the dataset
    if z_table is None:
        if train_graphs is not None:
            for graph in train_graphs:
                if not np.all(graph.nodes.species < num_species):
                    raise ValueError(
                        f"max(graph.nodes.species)={np.max(graph.nodes.species)} >= num_species={num_species}"
                    )
    else:
        if max(z_table.zs) >= num_species:
            raise ValueError(
                f"max(z_table.zs)={max(z_table.zs)} >= num_species={num_species}"
            )

    kwargs.update(
        dict(
            r_max=r_max,
            avg_num_neighbors=avg_num_neighbors,
            num_interactions=num_interactions,
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
        )
    )

    print(f"Create MACE with parameters {kwargs}")

    @hk.without_apply_rng
    @hk.transform
    def model_(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_z: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> jnp.ndarray:
        e3nn.config("path_normalization", path_normalization)
        e3nn.config("gradient_normalization", gradient_normalization)

        mace = GeneralMACE(output_irreps="0e", **kwargs)

        if hk.running_init():
            logging.info(
                "model: "
                f"num_features={mace.num_features} "
                f"hidden_irreps={mace.hidden_irreps} "
                f"sh_irreps={mace.sh_irreps} "
                f"interaction_irreps={mace.interaction_irreps} ",
            )

        contributions = mace(
            vectors, node_z, senders, receivers
        )  # [n_nodes, num_interactions, 0e]
        contributions = contributions.array[:, :, 0]  # [n_nodes, num_interactions]
        node_energies = jnp.sum(contributions, axis=1)  # [n_nodes, ]

        node_energies = mean + std * node_energies

        if learnable_atomic_energies:
            atomic_energies_ = hk.get_parameter(
                "atomic_energies",
                shape=(num_species,),
                init=hk.initializers.Constant(atomic_energies),
            )
        else:
            atomic_energies_ = jnp.asarray(atomic_energies)
        node_energies += atomic_energies_[node_z]  # [n_nodes, ]

        return node_energies

    if initialize_seed is not None and reload is None:
        params = jax.jit(model_.init)(
            jax.random.PRNGKey(initialize_seed),
            jnp.zeros((1, 3)),
            jnp.array([16]),
            jnp.array([0]),
            jnp.array([0]),
        )
    elif reload is not None:
        with open(f"{reload}/params.pkl", "rb") as f:
            params = pickle.load(f)
    else:
        params = None

    return model_.apply, params, num_interactions
    