from collections import namedtuple
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
from random import shuffle

import ase
import ase.data
import ase.io
import ase.spectrum.band_structure
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from .pad import pad_periodic_graph

from phonax.neighborhood import get_neighborhood
from phonax.data_utils import to_f32

Nodes = namedtuple(
    "Nodes",
    ["positions", "species", "mask_primitive", "v1", "v2", "index_cell0"],
)
Globals = namedtuple("Globals", ["kpt", "hessian",  "dynmat", "cell"]) 

cpu = jax.devices("cpu")[0]

try:
    gpu = jax.devices("gpu")[0]
except:
    gpu = None


def abs2(z: jnp.ndarray) -> jnp.ndarray:
    return z.real**2 + z.imag**2

#v1 = contraction_vector(r, ext_node_index_in_cell0, kpt, u1, m)
#def contraction_vector(r, i, k, u, m):
#    return u[i] * (np.exp(-1j * r @ k) / np.sqrt(m[i]))[:, None]  # [n_ext_nodes, 3]

def contraction_vector(r, i, k, u):
    return u[i] * (np.exp(-1j * r @ k))[:, None]  # [n_ext_nodes, 3]

def sign_sqrt(x):
    return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

def atoms_to_ext_graph(
    atoms: ase.Atoms, cutoff: float, num_message_passing: int
) -> jraph.GraphsTuple:
    senders, receivers, senders_unit_shifts = get_neighborhood(
        positions=atoms.positions,
        cutoff=cutoff,
        pbc=atoms.pbc,
        cell=atoms.cell.array,
    )

    num_atoms = len(atoms)
    kpt = atoms.info["k_point"] if "k_point" in atoms.info else np.array([0.0, 0.0, 0.0])
    dynmat = None
    atoms_hessian = None
    if "ifc" in atoms.info:
        # Molecular case
        atoms_ifc = atoms.info["ifc"]
        atoms_hessian = atoms.info["ifc_hessian"]
        u1 = jax.nn.one_hot(jnp.array([atoms_ifc[0]*3+atoms_ifc[1]]), 3*num_atoms).reshape((num_atoms,3)) # 1j * jnp.zeros((num_atoms,3))
        u2 = jax.nn.one_hot(jnp.array([atoms_ifc[2]*3+atoms_ifc[3]]), 3*num_atoms).reshape((num_atoms,3)) # 1j * jnp.zeros((num_atoms,3))
    # periodic crsytal cases
    else:
        u1 = atoms.arrays["eigvec0_re"] if "eigvec0_re" in atoms.arrays else None
        u2 = atoms.arrays["eigvec1_re"] if "eigvec1_re" in atoms.arrays else None
        dynmat = atoms.info["dynmatR"] + 1j * atoms.info["dynmatI"] if "dynmatR" in atoms.info and "dynmatI" in atoms.info else None

    (
        ext_node_positions,
        ext_node_index_in_cell0,
        ext_node_unit_shifts_from_cell0,
        ext_senders,
        ext_receivers,
    ) = pad_periodic_graph(
        atoms.positions,
        atoms.cell.array,
        senders,
        receivers,
        senders_unit_shifts,
        num_message_passing,
    )

    mask_primitive = np.all(
        ext_node_unit_shifts_from_cell0 == 0, axis=1
    )  # [n_ext_nodes, ]

    ext_node_species = atoms.numbers[ext_node_index_in_cell0]  # [n_ext_nodes, ]

    r = ext_node_positions
    #m = atoms.get_masses()
    
    if (kpt is not None) and (u1 is not None):
        v1 = contraction_vector(r, ext_node_index_in_cell0, kpt, u1)
    else:
        v1 = None
        
    if (kpt is not None) and (u2 is not None):
        v2 = contraction_vector(r, ext_node_index_in_cell0, kpt, u2)
    else:    
        v2 = None
    
    graph = jraph.GraphsTuple(
        nodes=Nodes(
            positions=ext_node_positions,
            species=ext_node_species,
            mask_primitive=mask_primitive,
            v1=v1,
            v2=v2,
            index_cell0=ext_node_index_in_cell0,
        ),
        edges=None,
        globals=Globals(
            kpt=kpt[None, :] if kpt is not None else None,
            dynmat = dynmat[None] if dynmat is not None else None,
            hessian=np.array([atoms_hessian])[None,:] if atoms_hessian is not None else None,
            cell=atoms.cell.array[None, :, :],
        ),
        receivers=ext_receivers,
        senders=ext_senders,
        n_edge=np.array([ext_senders.shape[0]]),
        n_node=np.array([ext_node_positions.shape[0]]),
    )
    return graph


class DataLoader:
    def __init__(
        self,
        graphs: List[jraph.GraphsTuple],
        n_node: int,
        n_edge: int,
        n_graph: int,
        shuffle: bool = True,
    ):
        self.graphs = graphs
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_graph = n_graph
        self.shuffle = shuffle
        self._length = None

    def __iter__(self):
        graphs = self.graphs.copy()  # this is a shallow copy
        if self.shuffle:
            shuffle(graphs)
        return jraph.dynamically_batch(
            graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
        )

    def __len__(self):
        if self.shuffle:
            return NotImplemented
        return self.approx_length()

    def approx_length(self):
        if self._length is None:
            self._length = 0
            for _ in self:
                self._length += 1
        return self._length



def sqrt(x):
    return jnp.sign(x) * jnp.sqrt(jnp.abs(x))



@partial(jax.jit, static_argnums=(1,))
def predict_hessian_matrix_spectra(
    w,
    model,  # model(relative_vectors, species, senders, receivers) -> [num_nodes]
    graph: jraph.GraphsTuple,
) -> jax.Array:
    """To be used with hessian_k"""

    def energy_fn(positions):
        vectors = positions[graph.receivers] - positions[graph.senders]
        node_energies = model(
            w, vectors, graph.nodes.species, graph.senders, graph.receivers
        )  # [n_nodes, ]

        node_energies = node_energies * graph.nodes.mask_primitive
        return jnp.sum(node_energies)

    basis = jnp.eye(
        graph.nodes.positions.size, dtype=graph.nodes.positions.dtype
    ).reshape(-1, *graph.nodes.positions.shape)

    def body_fn(i, hessian):
        return hessian.at[i].set(
            jax.jvp(
                jax.grad(energy_fn),
                (graph.nodes.positions,),
                (basis[i],),
            )[1]
        )

    hessian = jnp.zeros(
        (graph.nodes.positions.size,) + graph.nodes.positions.shape,
        dtype=graph.nodes.positions.dtype,
    )
    hessian = jax.lax.fori_loop(0, len(basis), body_fn, hessian)
    hessian = hessian.reshape(graph.nodes.positions.shape + graph.nodes.positions.shape)
    
    hessian = hessian.reshape((graph.nodes.positions.shape[0]*3,graph.nodes.positions.shape[0]*3))
    
    # Molecular case, with real hessian matrix
    hessian_spectra = jnp.sort(jnp.linalg.eigh(0.5*(hessian+hessian.T))[0])
    
    return hessian_spectra


@partial(jax.jit, static_argnums=(1,))
def predict_gamma_spectra(
    w,
    model,
    graph: jraph.GraphsTuple,
    masses,
) -> jax.Array:
    H = predict_hessian_matrix(
        w = w,
        model = model,
        graph = graph,
    )
    
    D_gamma = dynamical_matrix(
        kpt = jnp.zeros(3),
        graph = graph, 
        H = H, 
        masses = masses,
    )
        
    gamma_spectra = jnp.sort(jnp.linalg.eigh(0.5*(D_gamma+D_gamma.T))[0])
    
    return gamma_spectra
    
    
@partial(jax.jit, static_argnums=(1,))
def predict_gamma_spectra_filter(
    w,
    model,
    graph: jraph.GraphsTuple,
    masses,
) -> jax.Array:
    H = predict_hessian_matrix(
        w = w,
        model = model,
        graph = graph,
    )
    
    D_gamma = dynamical_matrix(
        kpt = jnp.zeros(3),
        graph = graph, 
        H = H, 
        masses = masses,
    )
    
    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass
        
    gamma_spectra = jnp.sqrt(jnp.sort(jnp.linalg.eigh(0.5*(D_gamma+D_gamma.T))[0])[6:])  * np.sqrt(conv_const) * hbar / cm_inv
    
    # unit cm^-1
    return gamma_spectra


@partial(jax.jit, static_argnums=(1,))
def predict_gamma_spectra_unfilter(
    w,
    model,
    graph: jraph.GraphsTuple,
    masses,
) -> jax.Array:
    H = predict_hessian_matrix(
        w = w,
        model = model,
        graph = graph,
    )
    
    D_gamma = dynamical_matrix(
        kpt = jnp.zeros(3),
        graph = graph, 
        H = H, 
        masses = masses,
    )
    
    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass
        
    gamma_spectra = jnp.sort(jnp.linalg.eigh(0.5*(D_gamma+D_gamma.T))[0]) 
    
    return gamma_spectra


@partial(jax.jit, static_argnums=(1,))
def predict_molecular_spectra_unfilter(
    w,
    model,
    graph: jraph.GraphsTuple,
    masses,
) -> jax.Array:
    H = predict_hessian_matrix(
        w = w,
        model = model,
        graph = graph,
    )
    
    D_gamma = dynamical_matrix(
        kpt = jnp.zeros(3),
        graph = graph, 
        H = H, 
        masses = masses,
    )
    
    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass
    
    full_spectra_info = jnp.linalg.eigh(0.5*(D_gamma.real+D_gamma.real.T))
        
    gamma_spectra = sign_sqrt(full_spectra_info[0])  * np.sqrt(conv_const) * hbar / cm_inv
    gamma_modes = full_spectra_info[1]
    gamma_Dk = 0.5*(D_gamma.real+D_gamma.real.T)
    
    return gamma_spectra, gamma_modes, gamma_Dk


@partial(jax.jit, static_argnums=(1,))
def predict_hessian_matrix(
    w,
    model,  # model(relative_vectors, species, senders, receivers) -> [num_nodes]
    graph: jraph.GraphsTuple,
) -> jax.Array:
    """To be used with hessian_k"""

    def energy_fn(positions):
        vectors = positions[graph.receivers] - positions[graph.senders]
        node_energies = model(
            w, vectors, graph.nodes.species, graph.senders, graph.receivers
        )  # [n_nodes, ]

        node_energies = node_energies * graph.nodes.mask_primitive
        return jnp.sum(node_energies)

    basis = jnp.eye(
        graph.nodes.positions.size, dtype=graph.nodes.positions.dtype
    ).reshape(-1, *graph.nodes.positions.shape)

    def body_fn(i, hessian):
        return hessian.at[i].set(
            jax.jvp(
                jax.grad(energy_fn),
                (graph.nodes.positions,),
                (basis[i],),
            )[1]
        )

    hessian = jnp.zeros(
        (graph.nodes.positions.size,) + graph.nodes.positions.shape,
        dtype=graph.nodes.positions.dtype,
    )
    hessian = jax.lax.fori_loop(0, len(basis), body_fn, hessian)
    return hessian.reshape(graph.nodes.positions.shape + graph.nodes.positions.shape)


# @partial(jax.jit, static_argnums=(0,))
def predict_hessian_vhv_products(
    model,  # model(relative_vectors, species, senders, receivers) -> [num_nodes]
    graph: jraph.GraphsTuple,
):
    num_graphs = len(graph.n_node)
    num_nodes = graph.nodes.positions.shape[0]

    def energy_fn(positions):
        vectors = positions[graph.receivers] - positions[graph.senders]
        node_energies = model(
            vectors, graph.nodes.species, graph.senders, graph.receivers
        )
        node_energies = node_energies * graph.nodes.mask_primitive
        assert node_energies.shape == (num_nodes,)

        return e3nn.scatter_sum(node_energies, nel=graph.n_node)

    x = graph.nodes.positions.astype(graph.nodes.v1.dtype)

    l = jnp.conj(graph.nodes.v1)
    r = graph.nodes.v2
    lhr = jax.jvp(lambda x: jax.jvp(energy_fn, (x,), (l,))[1], (x,), (r,))[1]
    assert lhr.shape == (num_graphs,)
    return lhr


@partial(jax.jit, static_argnames=("n_atoms",))
def hessian_k(kpt, graph, H, n_atoms: int):
    r"""Compute the Hessian matrix at a given k-point.

    .. math::

        \hat H_{ij}(\vec k) = \sum_a H_{0i,aj} e^{i \vec k \cdot (\vec x_{aj} - \vec x_{0i})}

    Args:
        kpt: k-point
        graph: extended graph
        H: Hessian matrix, computed with `predict_hessian_matrix`
    """
    r = graph.nodes.positions
    ph = jnp.exp(-1j * jnp.dot(r[:, None, :] - r[None, :, :], kpt))[:, None, :, None]
    a = graph.nodes.index_cell0
    i = jnp.arange(3)
    Hk = (
        jnp.zeros((n_atoms, 3, n_atoms, 3), dtype=ph.dtype)
        .at[jnp.ix_(a, i, a, i)]
        .add(ph * H)
    )
    return Hk.reshape((n_atoms * 3, n_atoms * 3))


@jax.jit
def dynamical_matrix(kpt, graph, H, masses):
    r"""Dynamical matrix at a given k-point.

    .. math::

        D_{ij}(\vec k) = \hat H_{ij}(\vec k) / \sqrt{m_i m_j}

    """
    Hk = hessian_k(kpt, graph, H, masses.size)
    Hk = Hk.reshape((masses.size, 3, masses.size, 3))

    iM = 1 / jnp.sqrt(masses)
    Hk = jnp.einsum("i,iujv,j->iujv", iM, Hk, iM)
    Hk = Hk.reshape((3 * masses.size, 3 * masses.size))
    return Hk


#@partial(jax.jit,device=gpu)
@jax.jit
def dynamical_matrix_parallel_k(all_kpts, graph, H, masses):

    print('print within dynamical_k function: compile')
    n_atoms = len(masses)
    num_ks = len(all_kpts)

    r = graph.nodes.positions
    #print(r.shape)
    #ph = jnp.exp(-1j * jnp.dot(r[:, None, :] - r[None, :, :], kpt))[:, None, :, None]
    a = graph.nodes.index_cell0
    #print(a.shape)
    i = jnp.arange(3)
    #print(i.shape)

    all_k = jnp.arange(len(all_kpts))
    #print(all_k.shape)

    ph = jnp.exp(-1j * jnp.einsum("ijk,ak->aij",r[:, None, :] - r[None, :, :],all_kpts))[:, :, None, :, None]
    #print(ph.shape)

    Hk0 = (
        jnp.zeros((num_ks,n_atoms, 3, n_atoms, 3), dtype=ph.dtype)
        .at[jnp.ix_(all_k,a, i, a, i)]
        .add(ph * H[None,:,:,:,:])
    )

    iM = 1 / jnp.sqrt(masses)
    Hk = jnp.einsum("i,aiujv,j->aiujv", iM, Hk0, iM)
    Hk = Hk.reshape((num_ks,3 * masses.size, 3 * masses.size))

    return  jnp.linalg.eigh(Hk)

def plot_bands(
    atoms, 
    graph, 
    hessian, 
    npoints=1000,
):
    # create ase cell object
    cell = atoms.cell.array  # [3, 3]
    cell = ase.Atoms(cell=cell, pbc=True).cell

    masses = ase.data.atomic_masses[atoms.get_atomic_numbers()]

    rec_vecs = 2 * np.pi * cell.reciprocal().real
    mp_band_path = cell.bandpath(npoints=npoints)

    all_kpts = mp_band_path.kpts @ rec_vecs
    all_eigs = []

    for kpt in tqdm(all_kpts):
        Dk = dynamical_matrix(kpt, graph, hessian, masses)
        Dk = np.asarray(Dk)
        all_eigs.append(np.sort(sqrt(np.linalg.eigh(Dk)[0])))

    all_eigs = np.stack(all_eigs)

    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass

    all_eigs = all_eigs * np.sqrt(conv_const) * hbar / cm_inv

    bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

    plt.figure(figsize=(7, 6), dpi=100)
    bs.plot(ax=plt.gca(), emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
    plt.ylabel("Phonon Frequency (cm$^{-1}$)")
    plt.tight_layout()
    return plt.gcf()

def plot_phonon_DOS(
    atoms, 
    graph, 
    hessian, 
    BZ_sampling = np.array([10,10,10]),
):
    # create ase cell object
    cell = atoms.cell.array  # [3, 3]
    cell = ase.Atoms(cell=cell, pbc=True).cell
    masses = ase.data.atomic_masses[atoms.get_atomic_numbers()]
    rec_vecs = 2 * np.pi * cell.reciprocal().real
    
    # BZ k point sampling
    k1_ = np.linspace(0., 1.0, BZ_sampling[0])
    k1_ = k1_[:-1]
    k2_ = np.linspace(0., 1.0, BZ_sampling[1])
    k2_ = k2_[:-1]
    k3_ = np.linspace(0., 1.0, BZ_sampling[2])
    k3_ = k3_[:-1]
    x, y, z = np.meshgrid(k1_, k2_, k3_, indexing='ij')
    BZ_k1 = x.reshape((-1,1))
    BZ_k2 = y.reshape((-1,1))
    BZ_k3 = z.reshape((-1,1))
    BZ_kpts = np.hstack((BZ_k1,BZ_k2,BZ_k3))
    BZ_xyz_kpts = BZ_kpts @ rec_vecs
    BZknum_tot = BZ_xyz_kpts.shape[0]
    
    eV_to_J = 1.60218e-19
    angstrom_to_m = 1e-10
    atom_mass = 1.660599e-27  # kg
    hbar = 1.05457182e-34
    cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass
    
    all_eigen_es = []
    all_eigen_vecs = []
    for kpt in BZ_xyz_kpts:
        Dk = dynamical_matrix(kpt, graph, hessian, masses)
        Dk = np.asarray(Dk)
        Dk_eigen_es, Dk_eigen_vecs = np.linalg.eigh(Dk)
        all_eigen_es.append(sqrt(Dk_eigen_es)* np.sqrt(conv_const) * hbar / cm_inv)
        all_eigen_vecs.append(Dk_eigen_vecs)
        
    Dk_eigen_es = np.array(all_eigen_es)
    Dk_eigen_vecs = np.array(all_eigen_vecs)
    Dk_eigen_vecs_proj = np.abs(Dk_eigen_vecs)**2

    es_min = np.min(Dk_eigen_es)
    es_max = np.max(Dk_eigen_es)

    config_symbols = atoms.get_chemical_symbols()
    config_symbols_types = set(config_symbols)

    es_scan = np.linspace(np.min([0,es_min])-20.0,es_max+20.0,1001)
    es_sigma = 2.0

    ph_dos = 0*es_scan
    ph_dos_proj = {}

    for ele_type in config_symbols_types:
        ph_dos_proj[ele_type] = 0*es_scan
        
    for idx, es_now in enumerate(es_scan):
        es_tmp = np.exp(-(Dk_eigen_es-es_now)**2/2/(es_sigma**2))/np.sqrt(2*np.pi*(es_sigma**2))
        for ele_type in config_symbols_types:
            type_check = np.repeat([ele == ele_type for ele in config_symbols],3)
            proj_vecs = np.sum(Dk_eigen_vecs_proj[:,type_check,:],axis=1)
            ph_dos_proj[ele_type][idx] = np.sum(es_tmp*proj_vecs)/BZknum_tot
        ph_dos[idx] = np.sum(es_tmp)/BZknum_tot
    
    plt.figure(figsize=(7, 6), dpi=100)
    plt.plot(es_scan,ph_dos,c = 'k',linewidth = 4)
    for ele_type in config_symbols_types:
        plt.plot(es_scan,ph_dos_proj[ele_type],linewidth = 2)
    plt_dos_label = ['Total phonon DOS']
    for ele_type in config_symbols_types:
        plt_dos_label.append(ele_type +'-projected DOS')
    plt.legend(plt_dos_label,fontsize= 11)
    plt.ylabel('Phonon DOS',size= 20)
    plt.xlabel("Phonon Frequency (cm$^{-1}$)",size=20)
    plt.xlim([es_scan[0],es_scan[-1]])
    plt.ylim([0,np.max(ph_dos)*1.1])
    plt.tight_layout()
    return plt.gcf()

def crystal_struct_to_graph(crystal_fname, r_max, num_message_passing): 
    predict_crystal = ase.io.read(crystal_fname)
    graph = atoms_to_ext_graph(predict_crystal, r_max, num_message_passing)   
    graph = jax.tree_util.tree_map(to_f32, graph)
    atoms = predict_crystal
    return atoms, graph


