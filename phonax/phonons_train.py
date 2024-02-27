import logging
import pickle
import sys
import time
from collections import namedtuple
from glob import glob
from random import shuffle
from typing import Any, Callable, Dict, List, Optional, Tuple

import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import tqdm


from phonax.utils import (
    compute_mae,
    compute_rel_mae,
    compute_rmse,
    compute_rel_rmse,
    compute_q95,
    compute_c,
)

from phonax.data_utils import (
    GraphDataLoader,
)

from phonax.energy_force_train import (
    evaluate,
)

from phonax.phonons import (
    DataLoader,
)


def ph_evaluate(
    loader: DataLoader,
    phonon_predictor: Callable[[Any, jraph.GraphsTuple], jnp.ndarray],
    params: Any,
    name: str = None,
    periodic_crystal: bool = False,
    limit: int = None,
):
    preds = []
    targets = []
    n_graphs = 0

    p_bar = tqdm.tqdm(loader, desc=f"Evaluating {name}", total=loader.approx_length())
    for graph in p_bar:
        pred = phonon_predictor(params, graph)
        mask = jraph.get_graph_padding_mask(graph)
        pred = pred[mask]  # [n_graphs, 3]
        if not periodic_crystal:
            target = graph.globals.hessian
        else:
            target = graph.globals.dynmat
        target = target[mask].squeeze()
        
        #print('check ph pred',pred.shape, target.shape)
        
        #target = np.stack(
        #    [graph.globals.λ1, graph.globals.λ2, np.zeros_like(graph.globals.λ1)],
        #    axis=1,
        #)  # [n_graphs, 3]
         # [n_graphs, 3]
        preds.append(pred)
        targets.append(target)

        n_graphs += mask.sum()
        if limit is not None and n_graphs >= limit:
            break

        p_bar.set_postfix({"n_graphs": n_graphs})

    preds = np.concatenate(preds, axis=0)  # [n_graphs, 3]
    targets = np.concatenate(targets, axis=0)  # [n_graphs, 3]

    #eV_to_J = 1.60218e-19
    #angstrom_to_m = 1e-10
    #atom_mass = 1.660599e-27  # kg
    #hbar = 1.05457182e-34
    #cm_inv = (0.124e-3) * (1.60218e-19)  # in J
    #conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass

    #preds = np.sqrt(np.abs(preds) * conv_const) * hbar / cm_inv
    #targets = np.sqrt(np.abs(targets) * conv_const) * hbar / cm_inv

    return preds, targets


def hessian_train(
    energy_forces_stress_model: Callable,
    phonon_model: Callable,
    params: Dict[str, Any],
    EF_train_loader: DataLoader,
    ph_train_loader: DataLoader,
    gradient_transform: Any,
    optimizer_state: Dict[str, Any],
    steps_per_interval: int,
    #max_num_intervals: int,
    #logger: logging.Logger,
    loss_fn_warmup: Callable,
    loss_fn_hessian: Callable,
    ema_decay: Optional[float] = None,
    hessian_mixing_ratio: float = 0.5,
    warmup_num_intervals: int = 10,
    phtrain_num_intervals: int = 50,
):
    num_updates = 0
    ema_params = params

    logging.info("Started training")
    
    
    #loss_fn_warmup = loss_EF_stress()

    ###def ph_loss_fn(graph: jraph.GraphsTuple, predictions: jnp.ndarray) -> jnp.ndarray:
    ###    #λ1p, λ2p, zp = predictions[:, 0], predictions[:, 1], predictions[:, 2]
    ###    #ph_loss = ph_abs2(λ1p - graph.globals.λ1) + ph_abs2(λ2p - graph.globals.λ2) + ph_abs2(zp)
    ###    #return ph_loss
    ###    #return abs2(λ1p - graph.globals.λ1) + abs2(λ2p - graph.globals.λ2) + 2.0*abs2(zp)
    ###    return jnp.square(predictions - graph.globals.hessian.squeeze())
    
    @jax.jit
    def update_fn_warmup(
        params, optimizer_state, ema_params, num_updates: int, graph: jraph.GraphsTuple
    ) -> Tuple[float, Any, Any]:
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = jraph.get_graph_padding_mask(graph)  # [n_graphs,]
        loss, grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn_warmup(graph, energy_forces_stress_model(params, graph)) * mask)
        )(params)
        updates, optimizer_state = gradient_transform.update(
            grad, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        if ema_decay is not None:
            decay = jnp.minimum(ema_decay, (1 + num_updates) / (10 + num_updates))
            ema_params = jax.tree_util.tree_map(
                lambda x, y: x * decay + y * (1 - decay), ema_params, params
            )
        else:
            ema_params = params
        return loss, params, optimizer_state, ema_params
            
    @jax.jit
    def update_fn_phstage(
        params, optimizer_state, ema_params, num_updates: int, EF_graph: jraph.GraphsTuple, ph_graph: jraph.GraphsTuple
    ) -> Tuple[float, Any, Any]:
        # graph is assumed to be padded by jraph.pad_with_graphs
        EF_mask = jraph.get_graph_padding_mask(EF_graph)  # [n_graphs,]
        ph_mask = jraph.get_graph_padding_mask(ph_graph)  # [n_graphs,]
        
        EF_loss, EF_grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn_warmup(EF_graph, energy_forces_stress_model(params, EF_graph)) * EF_mask)
        )(params)
        
        ph_loss, ph_grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn_hessian(ph_graph, phonon_model(params, ph_graph)) * ph_mask)
        )(params)
        
        grad = jax.tree_util.tree_map(
            #lambda x, y : x * (1.0-hessian_mixing_ratio) + y * hessian_mixing_ratio , EF_grad, ph_grad
            lambda x, y : x * 1.0 + y * 5.0 , EF_grad, ph_grad
        )
        
        loss = EF_loss * (1.0-hessian_mixing_ratio) + ph_loss * hessian_mixing_ratio
        
        updates, optimizer_state = gradient_transform.update(
            grad, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        if ema_decay is not None:
            decay = jnp.minimum(ema_decay, (1 + num_updates) / (10 + num_updates))
            ema_params = jax.tree_util.tree_map(
                lambda x, y: x * decay + y * (1 - decay), ema_params, params
            )
        else:
            ema_params = params
        return loss, params, optimizer_state, ema_params

    last_cache_size = update_fn_warmup._cache_size()


    def EF_interval_loader():
        i = 0
        while True:
            for graph in EF_train_loader:
                yield graph
                i += 1
                if i >= steps_per_interval:
                    return

    def ph_interval_loader():
        i = 0
        while True:
            for graph in ph_train_loader:
                yield graph
                i += 1
                if i >= steps_per_interval:
                    return

    warmup_stage = True
    # stage 1 warm up step
    for interval in range(warmup_num_intervals):
        yield interval, params, optimizer_state, ema_params, warmup_stage  

        # Train one epoch
        p_bar = tqdm.tqdm(
            EF_interval_loader(),
            desc=f"Interval {interval}",
            total=steps_per_interval,
        )
        for EF_graph in p_bar:
            num_updates += 1
            start_time = time.perf_counter()
            loss, params, optimizer_state, ema_params = update_fn_warmup(
                params, optimizer_state, ema_params, num_updates, EF_graph
            )
            loss = float(loss)
            step_time = time.perf_counter() - start_time

            #logger.log(
            #    {
            #        "loss": loss,
            #        "time": step_time,
            #        "mode": "opt",
            #        "num_updates": num_updates,
            #        "interval": interval,
            #        "interval_": num_updates / steps_per_interval,
            #    }
            #)

            p_bar.set_postfix({"loss": loss})

            if last_cache_size != update_fn_warmup._cache_size():
                last_cache_size = update_fn_warmup._cache_size()

                logging.info("Compiled function `update_fn` for args:")
                #logging.info(f"- n_node={graph.n_node} total={graph.n_node.sum()}")
                #logging.info(f"- n_edge={graph.n_edge} total={graph.n_edge.sum()}")
                logging.info(f"Outout: loss= {loss:.3f}")
                logging.info(
                    f"Compilation time: {step_time:.3f}s, cache size: {last_cache_size}"
                )
    logging.info("Done warmup stage")

    last_cache_size = update_fn_phstage._cache_size()
    warmup_stage = False
    # stage 2 training
    for interval in range(phtrain_num_intervals):
        yield interval, params, optimizer_state, ema_params, warmup_stage

        # Train one epoch
        p_bar = tqdm.tqdm(
            zip(EF_interval_loader(),ph_interval_loader()),
            desc=f"Interval {interval}",
            total=steps_per_interval,
        )
        for (EF_graph,ph_graph) in p_bar:
            num_updates += 1
            start_time = time.perf_counter()
            loss, params, optimizer_state, ema_params = update_fn_phstage(
                params, optimizer_state, ema_params, num_updates, EF_graph,ph_graph
            )
            loss = float(loss)
            step_time = time.perf_counter() - start_time

            #logger.log(
            #    {
            #        "loss": loss,
            #        "time": step_time,
            #        "mode": "opt",
            #        "num_updates": num_updates,
            #        "interval": interval,
            #        "interval_": num_updates / steps_per_interval,
            #    }
            #)

            p_bar.set_postfix({"loss": loss})

            if last_cache_size != update_fn_phstage._cache_size():
                last_cache_size = update_fn_phstage._cache_size()

                logging.info("Compiled function `update_fn` for args:")
                #logging.info(f"- n_node={graph.n_node} total={graph.n_node.sum()}")
                #logging.info(f"- n_edge={graph.n_edge} total={graph.n_edge.sum()}")
                logging.info(f"Outout: loss= {loss:.3f}")
                logging.info(
                    f"Compilation time: {step_time:.3f}s, cache size: {last_cache_size}"
                )
    logging.info("Done 2nd stage training with Hessian data")



def two_stage_hessian_train(
    energy_forces_stress_predictor,
    phonon_predictor,
    params,
    gradient_transform,
    optimizer_state,
    steps_per_interval: int,
    #max_num_intervals: int,
    EF_loss_fn: Callable,
    H_loss_fn: Callable,
    EF_train_loader,
    EF_valid_loader,
    H_train_loader,
    H_valid_loader,
    # hessian_loss_fn: = ,
    ema_decay: float = 0.99,
    EF_eval_train: bool = True,
    log_errors: str = "PerAtomMAE",
    hessian_mixing_ratio: float = 0.5,
    warmup_num_intervals: int = 10,
    phtrain_num_intervals: int = 50,
    periodic_crystal: bool = False,
    **kwargs,
):
    
    
    lowest_loss = np.inf
    #patience_counter = 0
    start_time = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []

    
    for interval, params, optimizer_state, ema_params, warmup_stage  in hessian_train(
        energy_forces_stress_model = energy_forces_stress_predictor,
        phonon_model = phonon_predictor,
        params = params,
        EF_train_loader = EF_train_loader,
        ph_train_loader = H_train_loader,
        gradient_transform = gradient_transform,
        optimizer_state = optimizer_state,
        steps_per_interval = steps_per_interval,
        loss_fn_warmup = EF_loss_fn,
        loss_fn_hessian = H_loss_fn,
        ema_decay = ema_decay,
        hessian_mixing_ratio = hessian_mixing_ratio,
        warmup_num_intervals = warmup_num_intervals,
        phtrain_num_intervals = phtrain_num_intervals,
    ):
        #continue
        #if interval % 1 != 0:
        #    continue
        
        # from phonon hessian training
        #with open(f"{directory}/{tag}.pkl", "wb") as f:
        #    pickle.dump(
        #        {
        #            "interval": interval,
        #            "ema_params": ema_params,
        #            "gin": gin.operative_config_str(),
        #        },
        #        f,
        #    )
        
        #with open(f"{directory}/{tag}.pkl", "wb") as f:
        #    pickle.dump(gin.operative_config_str(), f)
        #    pickle.dump(params, f)
        #    # pickle.dump(ema_params, f)

        # E / F evaluation
        def eval_and_print(loader, mode: str):
            loss_, metrics_ = evaluate(
                model=energy_forces_stress_predictor,
                params=ema_params,
                loss_fn=EF_loss_fn,
                data_loader=loader,
                name=mode,
            )
            metrics_["mode"] = mode
            metrics_["interval"] = interval
            #logger.log(metrics_)

            if log_errors == "PerAtomRMSE":
                error_e = "rmse_e_per_atom"
                error_f = "rmse_f"
                error_s = "rmse_s"
            elif log_errors == "rel_PerAtomRMSE":
                error_e = "rmse_e_per_atom"
                error_f = "rel_rmse_f"
                error_s = "rel_rmse_s"
            elif log_errors == "TotalRMSE":
                error_e = "rmse_e"
                error_f = "rmse_f"
                error_s = "rmse_s"
            elif log_errors == "PerAtomMAE":
                error_e = "mae_e_per_atom"
                error_f = "mae_f"
                error_s = "mae_s"
            elif log_errors == "rel_PerAtomMAE":
                error_e = "mae_e_per_atom"
                error_f = "rel_mae_f"
                error_s = "rel_mae_s"
            elif log_errors == "TotalMAE":
                error_e = "mae_e"
                error_f = "mae_f"
                error_s = "mae_s"

            def _(x: str):
                v: float = metrics_.get(x, None)
                if v is None:
                    return "N/A"
                if x.startswith("rel_"):
                    return f"{100 * v:.3f}%"
                if "_e" in x:
                    return f"{1e3 * v:.3f} meV"
                if "_f" in x:
                    return f"{1e3 * v:.3f} meV/Å"
                if "_s" in x:
                    return f"{1e3 * v:.3f} meV/Å³"
                raise NotImplementedError

            logging.info(
                f"Interval {interval}: {mode}: "
                f"loss={loss_:.4f}, "
                f"{error_e}={_(error_e)}, "
                f"{error_f}={_(error_f)}, "
                f"{error_s}={_(error_s)}"
            )
            
            print(
                f"Interval {interval}: {mode}: "
                f"loss={loss_:.4f}, "
                f"{error_e}={_(error_e)}, "
                f"{error_f}={_(error_f)}, "
                f"{error_s}={_(error_s)}"
            )
            
            #logger.log(
            #    {
            #        "eval" : 'force-step',
            #        "mode": mode,
            #        "interval": interval,
            #        "energy MAE": _(error_e),
            #        "force MAE": _(error_f),
            #    }
            #)
            
            
            return loss_

        if EF_eval_train or last_interval:
            #if isinstance(eval_train, (int, float)):
            #    eval_and_print(train_loader.subset(eval_train), "eval_train")
            #else:
            #    eval_and_print(train_loader, "eval_train")
            eval_and_print(EF_train_loader, "eval_train")

        #if (
        #    (eval_test or last_interval)
        #    and test_loader is not None
        #    and len(test_loader) > 0
        #):
        #    eval_and_print(test_loader, "eval_test")

        if EF_valid_loader is not None and len(EF_valid_loader) > 0:
            loss_ = eval_and_print(EF_valid_loader, "eval_valid")

            #if loss_ >= lowest_loss:
            #    patience_counter += 1
            #    if patience is not None and patience_counter >= patience:
            #        logging.info(
            #            f"Stopping optimization after {patience_counter} intervals without improvement"
            #        )
            #        break
            #else:
            #    lowest_loss = loss_
            #    patience_counter = 0

        eval_time_per_interval += [time.perf_counter() - start_time]
        avg_time_per_interval = np.mean(total_time_per_interval[-3:])
        avg_eval_time_per_interval = np.mean(eval_time_per_interval[-3:])

        logging.info(
            f"Interval {interval}: Time per interval: {avg_time_per_interval:.1f}s, "
            f"among which {avg_eval_time_per_interval:.1f}s for evaluation."
        )
        
        
        
        if warmup_stage:
            continue
            
        # add the hessian/phonon evaluation for molecular structures
        def ph_print_eval(loader, name):
            if loader.approx_length() == 0:
                return

            preds, targets = ph_evaluate(loader, phonon_predictor, ema_params, name, periodic_crystal)

            logging.info(
                f"interval {interval}: "
                f"{name} hessian MAE {np.abs(preds[:] - targets[:]).mean():.3f} eV/A2 "
            )
            print(
                f"interval {interval}: "
                f"{name} hessian MAE {np.abs(preds[:] - targets[:]).mean():.3f} eV/A2 "
            )
            #logger.log(
            #    {
            #        "mode": name,
            #        "interval": interval,
            #        "hessian MAE": np.abs(preds[:] - targets[:]).mean().item(),
            #    }
            #)

        ph_print_eval(H_train_loader, "hessian train")
        ph_print_eval(H_valid_loader, "hessian valid")

        #if last_interval:
        #    break

    logging.info("Training complete")
    
    
