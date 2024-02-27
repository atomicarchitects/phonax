import functools
import math
from typing import Callable, Dict, Optional, Union, List
import numpy as np

import logging
import json
import pickle

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np


import jax.numpy as jnp

from e3nn_jax import Irreps, Irrep
from e3nn_jax import IrrepsArray
from e3nn_jax import FunctionalLinear
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct, FunctionalTensorProduct
from e3nn_jax.haiku import FullyConnectedTensorProduct, Linear


from jax.nn import initializers
from jax import tree_util
from jax import jit
from jax import vmap
from jax import tree_map


import operator
import jax.nn
import functools


from phonax.utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
    bessel_basis,
    soft_envelope,
)

from phonax.data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)

from .blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
)


from nequip_jax import NEQUIPLayerHaiku, filter_layers



partial = functools.partial


Array = jnp.ndarray

UnaryFn = Callable[[Array], Array]

f32 = jnp.float32



def prod(xs):
    """From e3nn_jax/util/__init__.py."""
    return functools.reduce(operator.mul, xs, 1)


def tp_path_exists(arg_in1, arg_in2, arg_out):
    """Check if a tensor product path is viable.
    This helper function is similar to the one used in:
    https://github.com/e3nn/e3nn
    """
    arg_in1 = Irreps(arg_in1).simplify()
    arg_in2 = Irreps(arg_in2).simplify()
    arg_out = Irrep(arg_out)

    for multiplicity_1, irreps_1 in arg_in1:
        for multiplicity_2, irreps_2 in arg_in2:
            if arg_out in irreps_1 * irreps_2:
                return True
    return False

def naive_broadcast_decorator(func):
    def wrapper(*args):
        leading_shape = jnp.broadcast_shapes(*(arg.shape[:-1] for arg in args))
        args = [jnp.broadcast_to(arg, leading_shape + (-1,)) for arg in args]
        f = func
        for _ in range(len(leading_shape)):
            f = jax.vmap(f)
        return f(*args)

    return wrapper



NONLINEARITY = {
    'none': lambda x: x,
    'relu': jax.nn.relu,
    'swish': jax.nn.swish,    #lambda x: BetaSwish()(x),
    'raw_swish': jax.nn.swish,
    'tanh': jax.nn.tanh,
    'sigmoid': jax.nn.sigmoid,
    'silu': jax.nn.silu,
}







def get_nonlinearity_by_name(name: str) -> UnaryFn:
    if name in NONLINEARITY:
        return NONLINEARITY[name]
    raise ValueError(f'Nonlinearity "{name}" not found.')



def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5





class NequIPConvolution(hk.Module):
    def __init__(
        self,
        *,
        hidden_irreps: Irreps,
        use_sc: bool,
        nonlinearities: Union[str, Dict[str, str]],
        radial_net_nonlinearity: str = 'raw_swish',
        radial_net_n_hidden: int = 32,
        radial_net_n_layers: int = 2,
        num_basis: int = 8,
        avg_num_neighbors: float = 1.,
        scalar_mlp_std: float = 4.0,
    ):
        super().__init__()
        #print('calling IPconv init')
        
        self.hidden_irreps = hidden_irreps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        self.avg_num_neighbors = avg_num_neighbors
        self.scalar_mlp_std = scalar_mlp_std
        
    def __call__(
        self,
        node_features: IrrepsArray,
        node_attributes: IrrepsArray,
        edge_sh: Array,
        edge_src: Array,
        edge_dst: Array,
        edge_embedded: Array
    ) -> IrrepsArray:
        # Convolution outline in NequIP is:
        # Linear on nodes
        # TP + aggregate
        # divide by average number of neighbors
        # Concatenation
        # Linear on nodes
        # Self-connection
        # Gate
        #print('calling IPconv call')
        
        irreps_scalars = []
        irreps_nonscalars = []
        irreps_gate_scalars = []

        # get scalar target irreps
        for multiplicity, irrep in self.hidden_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if (Irrep(irrep).l == 0 and tp_path_exists(node_features.irreps, edge_sh.irreps, irrep)):
                irreps_scalars += [(multiplicity, irrep)]

        irreps_scalars = Irreps(irreps_scalars)

        # get non-scalar target irreps
        for multiplicity, irrep in self.hidden_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if (Irrep(irrep).l > 0 and tp_path_exists(node_features.irreps, edge_sh.irreps, irrep)):
                irreps_nonscalars += [(multiplicity, irrep)]

        irreps_nonscalars = Irreps(irreps_nonscalars)

        # get gate scalar irreps
        if tp_path_exists(node_features.irreps, edge_sh.irreps, '0e'):
            gate_scalar_irreps_type = '0e'
        else:
            gate_scalar_irreps_type = '0o'

        for multiplicity, irreps in irreps_nonscalars:
            irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

        irreps_gate_scalars = Irreps(irreps_gate_scalars)

        # final layer output irreps are all three
        # note that this order is assumed by the gate function later, i.e.
        # scalars left, then gate scalar, then non-scalars
        h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

        # self-connection: TP between node features and node attributes
        # this can equivalently be seen as a matrix multiplication acting on
        # the node features where the weight matrix is indexed by the node
        # attributes (typically the chemical species), i.e. a linear transform
        # that is a function of the species of the central atom
        
        #print('check irreps',h_out_irreps,node_features.irreps,node_attributes.irreps)
        
        if self.use_sc:
            self_connection = FullyConnectedTensorProduct(h_out_irreps)(node_features, node_attributes)

        h = node_features

        # first linear, stays in current h-space
        h = Linear(node_features.irreps)(h)

        # map node features onto edges for tp
        edge_features = jax.tree_map(lambda x: x[edge_src], h)

        # we gather the instructions for the tp as well as the tp output irreps
        mode = 'uvu'
        trainable = 'True'
        irreps_after_tp = []
        instructions = []

        # iterate over both arguments, i.e. node irreps and edge irreps
        # if they have a valid TP path for any of the target irreps,
        # add to instructions and put in appropriate position
        # we use uvu mode (where v is a single-element sum) and weights will
        # be provide externally by the scalar MLP
        # this triple for loop is similar to the one used in e3nn and nequip
        for i, (mul_in1, irreps_in1) in enumerate(node_features.irreps):
            for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in h_out_irreps:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # we will likely have constructed irreps in a non-l-increasing order
        # so we sort them to be in a l-increasing order
        irreps_after_tp, p, _ = Irreps(irreps_after_tp).sort()

        # if we sort the target irreps, we will have to sort the instructions
        # acoordingly, using the permutation indices
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [(irreps_in1, irreps_in2, p[irreps_out], mode, trainable)]

        # TP between spherical harmonics embedding of the edge vector
        # Y_ij(\hat{r}) and neighboring node h_j, weighted on a per-element basis
        # by the radial network R(r_ij)
        tp = FunctionalTensorProduct(
            irreps_in1=edge_features.irreps,
            irreps_in2=edge_sh.irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions
        )

        # scalar radial network, number of output neurons is the total number of
        # tensor product paths, nonlinearity must have f(0)=0 and MLP must not
        # have biases
        n_tp_weights = 0

        # get output dim of radial MLP / number of TP weights
        for ins in tp.instructions:
            if ins.has_weight:
                n_tp_weights += prod(ins.path_shape)
                
        #print('check call in IPconv')
        #print([self.radial_net_n_hidden] * self.radial_net_n_layers + [n_tp_weights])

        # build radial MLP R(r) that maps from interatomic distances to TP weights
        # must not use bias to that R(0)=0
        fc = hk.nets.MLP(
            (self.radial_net_n_hidden,) * self.radial_net_n_layers + (n_tp_weights,),
            activation = get_nonlinearity_by_name(self.radial_net_nonlinearity),
            with_bias=False,
            w_init = hk.initializers.RandomNormal(stddev=np.sqrt(self.scalar_mlp_std/self.radial_net_n_hidden)),
            # w_init = hk.initializers.RandomNormal(stddev=1.0/self.scalar_mlp_std),
            #scalar_mlp_std=self.scalar_mlp_std
        )
        
        #print('fc',type(fc))
        #print('edge_embedded',type(edge_embedded),edge_embedded.shape,edge_embedded)

        # the TP weights (v dimension) are given by the FC
        weight = fc(edge_embedded)

        # tp between node features that have been mapped onto edges and edge RSH
        # weighted by FC weight, we vmap over the dimension of the edges
        edge_features = jax.vmap(tp.left_right)(weight, edge_features, edge_sh)
        # TODO: It's not great that e3nn_jax automatically upcasts internally,
        # but this would need to be fixed at the e3nn level.
        edge_features = jax.tree_map(lambda x: x.astype(h.dtype), edge_features)

        # aggregate edges onto nodes after tp using e3nn-jax's index_add
        h_type = h.dtype
        h = jax.tree_map(
            lambda x: e3nn.index_add(edge_dst, x, out_dim=h.shape[0]),
            edge_features
        )
        # TODO: Remove this once e3nn_jax doesn't upcast inputs.
        h = jax.tree_map(lambda x: x.astype(h_type), h)

        # normalize by the average (not local) number of neighbors
        h = h / self.avg_num_neighbors

        # second linear, now we create extra gate scalars by mapping to h-out
        h = Linear(h_out_irreps)(h)

        # self-connection, similar to a resnet-update that sums the output from
        # the TP to chemistry-weighted h
        if self.use_sc:
            h = h + self_connection

        # gate nonlinearity, applied to gate data, consisting of:
        # a) regular scalars,
        # b) gate scalars, and
        # c) non-scalars to be gated
        # in this order
        gate_fn = partial(
            e3nn.gate,
            even_act=get_nonlinearity_by_name(self.nonlinearities['e']),
            odd_act=get_nonlinearity_by_name(self.nonlinearities['o']),
            even_gate_act=get_nonlinearity_by_name(self.nonlinearities['e']),
            odd_gate_act=get_nonlinearity_by_name(self.nonlinearities['o'])
        )

        h = gate_fn(h)
        # TODO: Remove this once e3nn_jax doesn't upcast inputs.
        h = jax.tree_map(lambda x: x.astype(h_type), h)

        return h





class NequIPEnergyModel(hk.Module):
    def __init__(
        self,
        *,
        output_irreps : e3nn.Irreps,
        graph_net_steps: int,
        use_sc: bool,
        nonlinearities: Union[str, Dict[str, str]],
        hidden_irreps: str,
        max_ell: int = 3,
        num_basis: int = 8,
        r_max: float = 4.,
        num_species: int = None,
        avg_r_min: float = None,
        num_features: int = 7,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_net_nonlinearity: str = 'raw_swish',
        radial_net_n_hidden: int = 64,
        radial_net_n_layers: int = 2,
        radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        shift: float = 0.,
        scale: float = 1.,
        avg_num_neighbors: float = 1.,
        scalar_mlp_std: float = 4.0,
    ):
        super().__init__()
        #print('calling Nequip init')
        
        output_irreps = e3nn.Irreps(output_irreps)
        self.output_irreps = output_irreps

        self.num_features = num_features
        self.max_ell=max_ell
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.hidden_irreps = Irreps(hidden_irreps)
        self.num_species = num_species
        self.graph_net_steps = graph_net_steps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        self.scalar_mlp_std = scalar_mlp_std
        
        
        #self.node_embedding = LinearNodeEmbeddingBlock(
        #    self.num_species, self.num_features * self.hidden_irreps
        #)
        self.node_embedding = LinearNodeEmbeddingBlock(
            self.num_species, self.hidden_irreps
        )
        
        #print('self node embed', self.node_embedding ) #,self.num_features * self.hidden_irreps)
        
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
        r_max = jnp.float32(self.r_max)
        hidden_irreps = Irreps(self.hidden_irreps)
        
        edge_src = senders
        edge_dst = receivers
        embedding_irreps = Irreps(f'{self.num_species}x0e')
        
        
        #node_attrs = IrrepsArray(embedding_irreps, node_specie)
        node_attrs = self.node_embedding(node_specie).astype(
            vectors.dtype
        )
    

        # edge embedding
        lengths = safe_norm(vectors, axis=-1)
        
        #dR = vectors
        #scalar_dr_edge = space.distance(dR)
        #edge_sh = e3nn.spherical_harmonics(self.sh_irreps, vectors, normalize=False)
        edge_sh = e3nn.spherical_harmonics(self.sh_irreps,vectors / lengths[..., None],normalize=False,normalization="component")


        embedded_dr_edge = self.radial_embedding(lengths).array
        
        # embedding layer
        #print('model irreps in',node_attrs.irreps)
        h_node = Linear(irreps_out=Irreps(hidden_irreps))(node_attrs)
        
        

        # convolutions
        for _ in range(self.graph_net_steps):
            h_node = NequIPConvolution(
                hidden_irreps=hidden_irreps,
                use_sc=self.use_sc,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_basis=self.num_basis,
                avg_num_neighbors=self.avg_num_neighbors,
                scalar_mlp_std=self.scalar_mlp_std
            )(h_node,
              node_attrs,
              edge_sh,
              edge_src,
              edge_dst,
              embedded_dr_edge
             )

        # output block, two Linears that decay dimensions from h to h//2 to 1
        for mul, ir in h_node.irreps:
            if ir == Irrep('0e'):
                mul_second_to_final = mul // 2

        second_to_final_irreps = Irreps(f'{mul_second_to_final}x0e')
        final_irreps = Irreps('1x0e')

        h_node = Linear(irreps_out=second_to_final_irreps)(h_node)
        h_node = Linear(irreps_out=final_irreps)(h_node).array
        
        return h_node
    


def NequIP_JAXMD_model(
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
        json_model = {}
        nequip_model_setup = {}

        if train_graphs is None:
            z_table = None
        else:
            z_table = get_atomic_number_table_from_zs(
                z for graph in train_graphs for z in graph.nodes.species
            )
        logging.info(f"z_table= {z_table}")
        
        nequip_model_setup['z_table'] = z_table

        zs_for_json = [int(zs) for zs in z_table.zs]
        json_model['zs'] = zs_for_json

        if save_dir_name:
            with open(f"{save_dir_name}/z_table.json", "w") as f:
                json.dump(zs_for_json, f)

        if avg_num_neighbors == "average":
            avg_num_neighbors = compute_avg_num_neighbors(train_graphs)
            print(
                f"Compute the average number of neighbors: {avg_num_neighbors:.3f}"
            )
        else:
            print(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")
            
        nequip_model_setup['avg_num_neighbors'] = avg_num_neighbors

        if avg_r_min == "average":
            avg_r_min = compute_avg_min_neighbor_distance(train_graphs)
            print(f"Compute the average min neighbor distance: {avg_r_min:.3f}")
        elif avg_r_min is None:
            print("Do not normalize the radial basis (avg_r_min=None)")
        else:
            print(f"Use the average min neighbor distance: {avg_r_min:.3f}")
            
        nequip_model_setup['avg_r_min'] = avg_r_min

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
            
            
        nequip_model_setup['atomic_energies'] = atomic_energies

        #print(type(atomic_energies),atomic_energies)

        atomic_energies_json = [float(E0) for E0 in atomic_energies]
        if save_dir_name:
            with open(f"{save_dir_name}/atomic_energies.json", "w") as f:
                json.dump(atomic_energies_json, f) #  E0s.tolist()

        json_model['atomic_energies'] = atomic_energies_json

        if save_dir_name:
            with open(f"{save_dir_name}/model.json", "w") as f:
                json.dump(json_model, f) #  E0s.tolist()


        if scaling is None:
            mean, std = 0.0, 1.0
        else:
            mean, std = scaling(train_graphs, atomic_energies)
            logging.info(
                f"Scaling with {scaling.__qualname__}: mean={mean:.2f}, std={std:.2f}"
            )
            
        nequip_model_setup['mean'] = mean
        nequip_model_setup['std'] = std
        
        if save_dir_name:
            with open(f"{save_dir_name}/nequip_model_setup.pkl", "wb") as f:
                pickle.dump(nequip_model_setup, f) #  E0s.tolist()
        
    else:
        with open(f"{reload}/nequip_model_setup.pkl", "rb") as f:
            nequip_model_setup = pickle.load(f)
            
        if save_dir_name:
            with open(f"{save_dir_name}/nequip_model_setup.pkl", "wb") as f:
                pickle.dump(nequip_model_setup, f) #  E0s.tolist()
        
        z_table = nequip_model_setup['z_table']
        avg_num_neighbors = nequip_model_setup['avg_num_neighbors']
        avg_r_min = nequip_model_setup['avg_r_min']
        atomic_energies = nequip_model_setup['atomic_energies']
        mean = nequip_model_setup['mean']
        std = nequip_model_setup['std']
        
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
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
        )
    )
    print(f"Create NequIP (JAX-MD version) with parameters {kwargs}")

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

        nequip = NequIPEnergyModel(output_irreps="0e", **kwargs)

        if hk.running_init():
            logging.info(
                "model: "
                f"hidden_irreps={nequip.hidden_irreps} "
                f"sh_irreps={nequip.sh_irreps} ",
            )

        contributions = nequip(
            vectors, node_z, senders, receivers
        )  # [n_nodes, num_interactions, 0e]
        node_energies = contributions[:, 0]
        
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
    
    if (initialize_seed is not None) and reload is None:
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

    return model_.apply, params, kwargs['graph_net_steps']



class NequIPEnergyModel_MarioJAX(hk.Module):
    def __init__(
        self,
        *,
        output_irreps : e3nn.Irreps,
        graph_net_steps: int,
        use_sc: bool,
        nonlinearities: Union[str, Dict[str, str]],
        hidden_irreps: str,
        max_ell: int = 3,
        num_basis: int = 8,
        r_max: float = 4.,
        num_species: int = None,
        avg_r_min: float = None,
        num_features: int = 7,
        #radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_net_nonlinearity: str = 'raw_swish',
        radial_net_n_hidden: int = 64,
        radial_net_n_layers: int = 2,
        #radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        shift: float = 0.,
        scale: float = 1.,
        avg_num_neighbors: float = 1.,
        #scalar_mlp_std: float = 4.0,
    ):
        super().__init__()
        #print('calling Nequip init')
        
        output_irreps = e3nn.Irreps(output_irreps)
        self.output_irreps = output_irreps

        self.num_features = num_features
        self.max_ell=max_ell
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.hidden_irreps = Irreps(hidden_irreps)
        self.num_species = num_species
        self.graph_net_steps = graph_net_steps
        self.use_sc = use_sc
        #self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        #self.scalar_mlp_std = scalar_mlp_std
        self.num_layers = self.graph_net_steps
        
        
        
    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        
        vectors = e3nn.IrrepsArray("1o", vectors) / self.r_max

        # Embedding: since we have a single atom type, we don't need embedding
        # The node features are just ones and the species indices are all zeros
        #emb = self.param("embedding", jax.random.normal, (self.num_species, 32))
        emb = hk.get_parameter(
            name = "embedding",
            shape = (self.num_species, 32),
            init = hk.initializers.RandomNormal(stddev=1.0),   #jax.random.normal,
            # dtype=
        )
        
        features = e3nn.as_irreps_array(emb[node_specie])
        
        
        layers = filter_layers(
            [e3nn.Irreps(self.hidden_irreps)] * (self.num_layers) , 
            #+ [e3nn.Irreps("128x0e")],
            max_ell=3,
        )
        
        
        for irreps in layers:
            layer = NEQUIPLayerHaiku(
                even_activation = jax.nn.silu,
                odd_activation = jax.nn.tanh,
                gate_activation = jax.nn.silu,
                mlp_activation = jax.nn.silu,
                mlp_n_hidden=self.radial_net_n_hidden,
                mlp_n_layers=self.radial_net_n_layers,
                n_radial_basis=self.num_basis,
                avg_num_neighbors=self.avg_num_neighbors,
                #scalar_mlp_std=self.scalar_mlp_std
                num_species = self.num_species,
                max_ell = 3,
                output_irreps = irreps, #output_irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
            )
            features = layer(vectors, features, node_specie, senders, receivers)
        
        features = e3nn.haiku.Linear("16x0e")(features)
        features = e3nn.haiku.Linear("0e")(features)
        
        energy = features.array.squeeze(1)
        
        return energy

def NequIP_JAX_model(
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
    #path_normalization="path",
    #gradient_normalization="path",
    learnable_atomic_energies=False,
    save_dir_name = None,
    reload = None,
    #radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    #radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    **kwargs,
):
    
    if reload is None:
        json_model = {}
        nequip_model_setup = {}

        if train_graphs is None:
            z_table = None
        else:
            z_table = get_atomic_number_table_from_zs(
                z for graph in train_graphs for z in graph.nodes.species
            )
        logging.info(f"z_table= {z_table}")
        
        nequip_model_setup['z_table'] = z_table

        zs_for_json = [int(zs) for zs in z_table.zs]
        json_model['zs'] = zs_for_json

        if save_dir_name:
            with open(f"{save_dir_name}/z_table.json", "w") as f:
                json.dump(zs_for_json, f)

        if avg_num_neighbors == "average":
            avg_num_neighbors = compute_avg_num_neighbors(train_graphs)
            print(
                f"Compute the average number of neighbors: {avg_num_neighbors:.3f}"
            )
        else:
            print(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")
            
        nequip_model_setup['avg_num_neighbors'] = avg_num_neighbors

        if avg_r_min == "average":
            avg_r_min = compute_avg_min_neighbor_distance(train_graphs)
            print(f"Compute the average min neighbor distance: {avg_r_min:.3f}")
        elif avg_r_min is None:
            print("Do not normalize the radial basis (avg_r_min=None)")
        else:
            print(f"Use the average min neighbor distance: {avg_r_min:.3f}")
            
        nequip_model_setup['avg_r_min'] = avg_r_min

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
            
            
        nequip_model_setup['atomic_energies'] = atomic_energies

        #print(type(atomic_energies),atomic_energies)

        atomic_energies_json = [float(E0) for E0 in atomic_energies]
        if save_dir_name:
            with open(f"{save_dir_name}/atomic_energies.json", "w") as f:
                json.dump(atomic_energies_json, f) #  E0s.tolist()

        json_model['atomic_energies'] = atomic_energies_json

        if save_dir_name:
            with open(f"{save_dir_name}/model.json", "w") as f:
                json.dump(json_model, f) #  E0s.tolist()


        if scaling is None:
            mean, std = 0.0, 1.0
        else:
            mean, std = scaling(train_graphs, atomic_energies)
            logging.info(
                f"Scaling with {scaling.__qualname__}: mean={mean:.2f}, std={std:.2f}"
            )
            
        nequip_model_setup['mean'] = mean
        nequip_model_setup['std'] = std
        
        if save_dir_name:
            with open(f"{save_dir_name}/nequip_model_setup.pkl", "wb") as f:
                pickle.dump(nequip_model_setup, f) #  E0s.tolist()
        
    else:
        with open(f"{reload}/nequip_model_setup.pkl", "rb") as f:
            nequip_model_setup = pickle.load(f)
            
        if save_dir_name:
            with open(f"{save_dir_name}/nequip_model_setup.pkl", "wb") as f:
                pickle.dump(nequip_model_setup, f) #  E0s.tolist()
        
        z_table = nequip_model_setup['z_table']
        avg_num_neighbors = nequip_model_setup['avg_num_neighbors']
        avg_r_min = nequip_model_setup['avg_r_min']
        atomic_energies = nequip_model_setup['atomic_energies']
        mean = nequip_model_setup['mean']
        std = nequip_model_setup['std']
        
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
            avg_r_min=avg_r_min,
            num_species=num_species,
            #radial_basis=radial_basis,
            #radial_envelope=radial_envelope,
        )
    )
    
    print(f"Create NequIP (JAX version) with parameters {kwargs}")

    @hk.without_apply_rng
    @hk.transform
    def model_(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_z: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> jnp.ndarray:

        nequip = NequIPEnergyModel_MarioJAX(output_irreps="0e", **kwargs)

        if hk.running_init():
            logging.info(
                "model: "
                f"hidden_irreps={nequip.hidden_irreps} "
                f"sh_irreps={nequip.sh_irreps} ",
            )

        node_energies = nequip(
            vectors, node_z, senders, receivers
        )
        
        node_energies = mean + std * node_energies

        if learnable_atomic_energies:
            atomic_energies_ = hk.get_parameter(
                "atomic_energies",
                shape=(num_species,),
                init=hk.initializers.Constant(atomic_energies),
            )
        else:
            #print('energy shift')
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

    return model_.apply, params, kwargs['graph_net_steps']





