import json
import os
import pickle
import random
import sys
import pickle
import jraph
import tqdm

from typing import Callable, Dict, List, Optional
import haiku as hk
import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from ase import Atoms
import time

import matplotlib.pyplot as plt


import glob
import spglib


from phonax.utils import find_primitive_structure
from phonax.trained_models import (
    NequIP_JAXMD_uniIAP_model,
    NequIP_JAXMD_uniIAP_PBEsol_finetuned_model,
)
from phonax.phonons import (
    dynamical_matrix_parallel_k,
    dynamical_matrix,
    atoms_to_ext_graph,
    predict_hessian_matrix,
)
from phonax.data_utils import (
    to_f32,
)



eV_to_J = 1.60218e-19
angstrom_to_m = 1e-10
atom_mass = 1.660599e-27  # kg
hbar = 1.05457182e-34
cm_inv = (0.124e-3) * (1.60218e-19)  # in J
conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass
    
def sqrt(x):
    return jnp.sign(x) * jnp.sqrt(jnp.abs(x))



def predict_ph_v1(use_finetune,lat_reduce,gamma_only,cif_fname):
    cif_fname_path = cif_fname.name
    cif_phonon_pred_fname = cif_fname_path+'.phbands.png'  
    config = ase.io.read(cif_fname_path)
    print('number of atoms (original): ',len(config))
    if lat_reduce:
        config = find_primitive_structure(config)
        print('number of atoms (reduced): ',len(config))

    config_id = cif_fname_path.split('/')[-1].split('.')[0]

    if use_finetune:
        model_fn, params, num_message_passing, r_max = NequIP_JAXMD_uniIAP_PBEsol_finetuned_model(os.path.join(os.getcwd(), 'trained-models'))
    else:
        model_fn, params, num_message_passing, r_max = NequIP_JAXMD_uniIAP_model(os.path.join(os.getcwd(), 'trained-models'))

    graph = atoms_to_ext_graph(config, r_max, num_message_passing)

    graph = jax.tree_util.tree_map(to_f32, graph)

    stime = time.time()
    H = predict_hessian_matrix(params, model_fn, graph)
    print('hessian compute step, ', time.time()-stime)
    print('hessian shape', H.shape)


    # create ase cell object
    cell = graph.globals.cell[0]  # [3, 3]
    cell = ase.Atoms(cell=cell, pbc=True).cell

    masses = ase.data.atomic_masses[graph.nodes.species][graph.nodes.mask_primitive]
    iM = np.diag(1 / np.sqrt(masses))
    
    if not gamma_only:
        rec_vecs = 2 * np.pi * cell.reciprocal().real
        mp_band_path = cell.bandpath(npoints=200)

        all_kpts = mp_band_path.kpts @ rec_vecs
        all_eigs = []

        stime = time.time()
        all_eigs, all_eigen_vecs  = dynamical_matrix_parallel_k(all_kpts, graph, H, masses)
        all_eigs = sqrt(all_eigs) * np.sqrt(conv_const) * hbar / cm_inv
        all_eigs.block_until_ready()
        
        print('hessian_k computation step (bands), ', time.time()-stime)

        bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

        # BZ k point sampling
        scan_num = 11
        x_ = np.linspace(0., 1.0, scan_num)
        x_ = x_[:-1]
        y_ = np.linspace(0., 1.0, scan_num)
        y_ = y_[:-1]
        z_ = np.linspace(0., 1.0, scan_num)
        z_ = z_[:-1]

        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        BZ_k1 = x.reshape((-1,1))
        BZ_k2 = y.reshape((-1,1))
        BZ_k3 = z.reshape((-1,1))
        BZ_kpts = np.hstack((BZ_k1,BZ_k2,BZ_k3))

        BZ_xyz_kpts = BZ_kpts @ rec_vecs
        BZknum_tot = BZ_xyz_kpts.shape[0]

        #all_Hk = []
        all_eigen_es = []
        all_eigen_vecs = []
        stime = time.time()
        #for kpt in BZ_xyz_kpts:
        #    Hk = hessian_k(kpt, graph, H, masses)
        #    #all_Hk.append(Hk)
        #    Hk_eigen_es, Hk_eigen_vecs = np.linalg.eigh(Hk)
        #    all_eigen_es.append(sqrt(Hk_eigen_es)* np.sqrt(conv_const) * hbar / cm_inv)
        #    all_eigen_vecs.append(Hk_eigen_vecs)
        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        all_eigen_es, all_eigen_vecs  = dynamical_matrix_parallel_k(BZ_xyz_kpts, graph, H, masses)
        all_eigen_es = sqrt(all_eigen_es) * np.sqrt(conv_const) * hbar / cm_inv
        all_eigen_es.block_until_ready()
        all_eigen_vecs.block_until_ready()

        print('hessian_k computation step (DOS), ', time.time()-stime)
        stime = time.time()
        Hk_eigen_es = np.array(all_eigen_es)
        Hk_eigen_vecs = np.array(all_eigen_vecs)

        Hk_eigen_vecs_proj = np.abs(Hk_eigen_vecs)**2

        es_min = np.min(Hk_eigen_es)
        es_max = np.max(Hk_eigen_es)

        config_symbols = config.get_chemical_symbols()
        config_symbols_types = set(config_symbols)

        num_es = 501
        es_scan = np.linspace(np.min([0,es_min])-20.0,es_max+20.0,num_es)
        es_sigma = ( es_max - es_min ) / num_es * 2.0

        ph_dos = 0*es_scan
        ph_dos_proj = {}

        for ele_type in config_symbols_types:
            ph_dos_proj[ele_type] = 0*es_scan

        print('proj-pdos step1, ', time.time()-stime)
        stime = time.time()
        for idx, es_now in enumerate(es_scan):
            es_tmp = np.exp(-(Hk_eigen_es-es_now)**2/2/(es_sigma**2))/np.sqrt(2*np.pi*(es_sigma**2))

            for ele_type in config_symbols_types:

                type_check = np.repeat([ele == ele_type for ele in config_symbols],3)
                #print(type_check.shape)
                #print(Hk_eigen_vecs_proj.shape)

                proj_vecs = np.sum(Hk_eigen_vecs_proj[:,type_check,:],axis=1)
                ph_dos_proj[ele_type][idx] = np.sum(es_tmp*proj_vecs)/BZknum_tot

            ph_dos[idx] = np.sum(es_tmp)/BZknum_tot

        print('proj-pdos step2, ', time.time()-stime)
        #plt.figure(figsize=(10, 19), dpi=200)
        fig, axs = plt.subplots(2,figsize=(7, 10),dpi=100)

        bs.plot(ax=axs[0], emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
        axs[0].set_ylabel("Phonon Frequency (cm$^{-1}$)",size= 20)
        axs[0].set_title(config_id)



        axs[1].plot(es_scan,ph_dos,c = 'k',linewidth = 4)
        for ele_type in config_symbols_types:
            axs[1].plot(es_scan,ph_dos_proj[ele_type],linewidth = 2)
        #plt.plot(es_scan,ph_dos_proj['Ba'])
        #plt.plot(es_scan,ph_dos_proj['Te'])

        plt_dos_label = ['Total phonon DOS']

        for ele_type in config_symbols_types:
            plt_dos_label.append(ele_type +'-projected DOS')

        axs[1].legend(plt_dos_label,fontsize= 11)
        axs[1].set_ylabel('Phonon DOS',size= 20)
        axs[1].set_xlabel("Phonon Frequency (cm$^{-1}$)",size= 20)
        axs[1].set_xlim([es_scan[0],es_scan[-1]])
        axs[1].set_ylim([0,np.max(ph_dos)*1.1])

        plt.tight_layout()
        plt.savefig(cif_phonon_pred_fname)
        
    else:
        BZknum_tot = 1
        stime = time.time()

        Hk = dynamical_matrix(np.array([0.0,0.0,0.0]), graph, H, masses)
        Hk_eigen_es, Hk_eigen_vecs = np.linalg.eigh(Hk)
        all_eigen_es = sqrt(Hk_eigen_es)* np.sqrt(conv_const) * hbar / cm_inv
        all_eigen_vecs = Hk_eigen_vecs

        print('hessian_k2 step, ', time.time()-stime)
        #all_Hk = np.array(all_Hk)
        Hk_eigen_es = np.array(all_eigen_es)
        Hk_eigen_vecs = np.array(all_eigen_vecs)

        Hk_eigen_vecs_proj = np.abs(Hk_eigen_vecs)**2

        es_min = np.min(Hk_eigen_es)
        es_max = np.max(Hk_eigen_es)

        config_symbols = config.get_chemical_symbols()
        config_symbols_types = set(config_symbols)

        num_es = 501
        es_scan = np.linspace(np.min([0,es_min])-20.0,es_max+20.0,num_es)
        es_sigma = ( es_max - es_min ) / num_es * 2.0

        ph_dos = 0*es_scan
        ph_dos_proj = {}

        for ele_type in config_symbols_types:
            ph_dos_proj[ele_type] = 0*es_scan


        for idx, es_now in enumerate(es_scan):
            es_tmp = np.exp(-(Hk_eigen_es-es_now)**2/2/(es_sigma**2))/np.sqrt(2*np.pi*(es_sigma**2))

            for ele_type in config_symbols_types:

                type_check = np.repeat([ele == ele_type for ele in config_symbols],3)
                #print(type_check.shape)
                #print(Hk_eigen_vecs_proj.shape)

                proj_vecs = np.sum(Hk_eigen_vecs_proj[type_check,:],axis=0)
                ph_dos_proj[ele_type][idx] = np.sum(es_tmp*proj_vecs)/BZknum_tot

            ph_dos[idx] = np.sum(es_tmp)/BZknum_tot


        plt.figure(figsize=(7,6), dpi=100)
        #fig, axs = plt.subplots(2,figsize=(7, 10),dpi=100)
        plt.plot(es_scan,ph_dos,c = 'k',linewidth = 4)
        for ele_type in config_symbols_types:
            plt.plot(es_scan,ph_dos_proj[ele_type],linewidth = 2)

        plt_dos_label = ['Total phonon DOS']

        for ele_type in config_symbols_types:
            plt_dos_label.append(ele_type +'-projected DOS')

        plt.legend(plt_dos_label,fontsize= 11)
        plt.ylabel('Phonon DOS',size= 20)
        plt.xlabel("Phonon Frequency (cm$^{-1}$)",size= 20)
        plt.xlim([es_scan[0],es_scan[-1]])
        plt.ylim([0,np.max(ph_dos)*1.1])

        plt.tight_layout()
        plt.savefig(cif_phonon_pred_fname)
    
    return cif_phonon_pred_fname


def predict_ph_v2(use_finetune, egnn_model, lat_reduce, gamma_only, cif_fname):
    cif_fname_path = cif_fname.name
    cif_phonon_pred_fname = cif_fname_path+'.phbands.png'  
    config = ase.io.read(cif_fname_path)
    print('number of atoms (original): ',len(config))
    if lat_reduce:
        config = find_primitive_structure(config)
        print('number of atoms (reduced): ',len(config))

    config_id = cif_fname_path.split('/')[-1].split('.')[0]

    if use_finetune:
        model_fn, params, num_message_passing, r_max = NequIP_JAXMD_uniIAP_PBEsol_finetuned_model()
    else:
        model_fn, params, num_message_passing, r_max = NequIP_JAXMD_uniIAP_model()

    graph = atoms_to_ext_graph(config, r_max, num_message_passing)

    graph = jax.tree_util.tree_map(to_f32, graph)

    stime = time.time()
    H = predict_hessian_matrix(params, model_fn, graph)
    print('hessian compute step, ', time.time()-stime)
    print('hessian shape', H.shape)


    # create ase cell object
    cell = graph.globals.cell[0]  # [3, 3]
    cell = ase.Atoms(cell=cell, pbc=True).cell

    masses = ase.data.atomic_masses[graph.nodes.species][graph.nodes.mask_primitive]
    iM = np.diag(1 / np.sqrt(masses))
    
    if not gamma_only:
        rec_vecs = 2 * np.pi * cell.reciprocal().real
        mp_band_path = cell.bandpath(npoints=200)

        all_kpts = mp_band_path.kpts @ rec_vecs
        all_eigs = []

        stime = time.time()
        all_eigs, all_eigen_vecs  = dynamical_matrix_parallel_k(all_kpts, graph, H, masses)
        all_eigs = sqrt(all_eigs) * np.sqrt(conv_const) * hbar / cm_inv
        all_eigs.block_until_ready()
        
        print('hessian_k computation step (bands), ', time.time()-stime)

        bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

        # BZ k point sampling
        scan_num = 11
        x_ = np.linspace(0., 1.0, scan_num)
        x_ = x_[:-1]
        y_ = np.linspace(0., 1.0, scan_num)
        y_ = y_[:-1]
        z_ = np.linspace(0., 1.0, scan_num)
        z_ = z_[:-1]

        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        BZ_k1 = x.reshape((-1,1))
        BZ_k2 = y.reshape((-1,1))
        BZ_k3 = z.reshape((-1,1))
        BZ_kpts = np.hstack((BZ_k1,BZ_k2,BZ_k3))

        BZ_xyz_kpts = BZ_kpts @ rec_vecs
        BZknum_tot = BZ_xyz_kpts.shape[0]

        #all_Hk = []
        all_eigen_es = []
        all_eigen_vecs = []
        stime = time.time()
        #for kpt in BZ_xyz_kpts:
        #    Hk = hessian_k(kpt, graph, H, masses)
        #    #all_Hk.append(Hk)
        #    Hk_eigen_es, Hk_eigen_vecs = np.linalg.eigh(Hk)
        #    all_eigen_es.append(sqrt(Hk_eigen_es)* np.sqrt(conv_const) * hbar / cm_inv)
        #    all_eigen_vecs.append(Hk_eigen_vecs)
        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        all_eigen_es, all_eigen_vecs  = dynamical_matrix_parallel_k(BZ_xyz_kpts, graph, H, masses)
        all_eigen_es = sqrt(all_eigen_es) * np.sqrt(conv_const) * hbar / cm_inv
        all_eigen_es.block_until_ready()
        all_eigen_vecs.block_until_ready()

        print('hessian_k computation step (DOS), ', time.time()-stime)
        stime = time.time()
        Hk_eigen_es = np.array(all_eigen_es)
        Hk_eigen_vecs = np.array(all_eigen_vecs)

        Hk_eigen_vecs_proj = np.abs(Hk_eigen_vecs)**2

        es_min = np.min(Hk_eigen_es)
        es_max = np.max(Hk_eigen_es)

        config_symbols = config.get_chemical_symbols()
        config_symbols_types = set(config_symbols)

        num_es = 501
        es_scan = np.linspace(np.min([0,es_min])-20.0,es_max+20.0,num_es)
        es_sigma = ( es_max - es_min ) / num_es * 2.0

        ph_dos = 0*es_scan
        ph_dos_proj = {}

        for ele_type in config_symbols_types:
            ph_dos_proj[ele_type] = 0*es_scan

        print('proj-pdos step1, ', time.time()-stime)
        stime = time.time()
        for idx, es_now in enumerate(es_scan):
            es_tmp = np.exp(-(Hk_eigen_es-es_now)**2/2/(es_sigma**2))/np.sqrt(2*np.pi*(es_sigma**2))

            for ele_type in config_symbols_types:

                type_check = np.repeat([ele == ele_type for ele in config_symbols],3)
                #print(type_check.shape)
                #print(Hk_eigen_vecs_proj.shape)

                proj_vecs = np.sum(Hk_eigen_vecs_proj[:,type_check,:],axis=1)
                ph_dos_proj[ele_type][idx] = np.sum(es_tmp*proj_vecs)/BZknum_tot

            ph_dos[idx] = np.sum(es_tmp)/BZknum_tot

        print('proj-pdos step2, ', time.time()-stime)
        #plt.figure(figsize=(10, 19), dpi=200)
        fig, axs = plt.subplots(2,figsize=(7, 10),dpi=100)

        bs.plot(ax=axs[0], emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
        axs[0].set_ylabel("Phonon Frequency (cm$^{-1}$)",size= 20)
        axs[0].set_title(config_id)



        axs[1].plot(es_scan,ph_dos,c = 'k',linewidth = 4)
        for ele_type in config_symbols_types:
            axs[1].plot(es_scan,ph_dos_proj[ele_type],linewidth = 2)
        #plt.plot(es_scan,ph_dos_proj['Ba'])
        #plt.plot(es_scan,ph_dos_proj['Te'])

        plt_dos_label = ['Total phonon DOS']

        for ele_type in config_symbols_types:
            plt_dos_label.append(ele_type +'-projected DOS')

        axs[1].legend(plt_dos_label,fontsize= 11)
        axs[1].set_ylabel('Phonon DOS',size= 20)
        axs[1].set_xlabel("Phonon Frequency (cm$^{-1}$)",size= 20)
        axs[1].set_xlim([es_scan[0],es_scan[-1]])
        axs[1].set_ylim([0,np.max(ph_dos)*1.1])

        plt.tight_layout()
        plt.savefig(cif_phonon_pred_fname)
        
    else:
        BZknum_tot = 1
        stime = time.time()

        Hk = dynamical_matrix(np.array([0.0,0.0,0.0]), graph, H, masses)
        Hk_eigen_es, Hk_eigen_vecs = np.linalg.eigh(Hk)
        all_eigen_es = sqrt(Hk_eigen_es)* np.sqrt(conv_const) * hbar / cm_inv
        all_eigen_vecs = Hk_eigen_vecs

        print('hessian_k2 step, ', time.time()-stime)
        #all_Hk = np.array(all_Hk)
        Hk_eigen_es = np.array(all_eigen_es)
        Hk_eigen_vecs = np.array(all_eigen_vecs)

        Hk_eigen_vecs_proj = np.abs(Hk_eigen_vecs)**2

        es_min = np.min(Hk_eigen_es)
        es_max = np.max(Hk_eigen_es)

        config_symbols = config.get_chemical_symbols()
        config_symbols_types = set(config_symbols)

        num_es = 501
        es_scan = np.linspace(np.min([0,es_min])-20.0,es_max+20.0,num_es)
        es_sigma = ( es_max - es_min ) / num_es * 2.0

        ph_dos = 0*es_scan
        ph_dos_proj = {}

        for ele_type in config_symbols_types:
            ph_dos_proj[ele_type] = 0*es_scan


        for idx, es_now in enumerate(es_scan):
            es_tmp = np.exp(-(Hk_eigen_es-es_now)**2/2/(es_sigma**2))/np.sqrt(2*np.pi*(es_sigma**2))

            for ele_type in config_symbols_types:

                type_check = np.repeat([ele == ele_type for ele in config_symbols],3)
                #print(type_check.shape)
                #print(Hk_eigen_vecs_proj.shape)

                proj_vecs = np.sum(Hk_eigen_vecs_proj[type_check,:],axis=0)
                ph_dos_proj[ele_type][idx] = np.sum(es_tmp*proj_vecs)/BZknum_tot

            ph_dos[idx] = np.sum(es_tmp)/BZknum_tot


        plt.figure(figsize=(7,6), dpi=100)
        #fig, axs = plt.subplots(2,figsize=(7, 10),dpi=100)
        plt.plot(es_scan,ph_dos,c = 'k',linewidth = 4)
        for ele_type in config_symbols_types:
            plt.plot(es_scan,ph_dos_proj[ele_type],linewidth = 2)

        plt_dos_label = ['Total phonon DOS']

        for ele_type in config_symbols_types:
            plt_dos_label.append(ele_type +'-projected DOS')

        plt.legend(plt_dos_label,fontsize= 11)
        plt.ylabel('Phonon DOS',size= 20)
        plt.xlabel("Phonon Frequency (cm$^{-1}$)",size= 20)
        plt.xlim([es_scan[0],es_scan[-1]])
        plt.ylim([0,np.max(ph_dos)*1.1])

        plt.tight_layout()
        plt.savefig(cif_phonon_pred_fname)
    
    return cif_phonon_pred_fname
 




