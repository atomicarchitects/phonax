import itertools
import logging
import time
import pickle
import yaml
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import tqdm

from .utils import (
    compute_mae,
    compute_rel_mae,
    compute_rmse,
    compute_rel_rmse,
    compute_q95,
    compute_c,
)

from .data_utils import (
    GraphDataLoader,
)


def train(
    model: Callable,
    params: Dict[str, Any],
    loss_fn: Any,
    train_loader: GraphDataLoader,
    gradient_transform: Any,
    optimizer_state: Dict[str, Any],
    steps_per_interval: int,
    ema_decay: Optional[float] = None,
):
    num_updates = 0
    ema_params = params

    #logging.info("Started training")
    print("Started training")

    @jax.jit
    def update_fn(
        params, optimizer_state, ema_params, num_updates: int, graph: jraph.GraphsTuple
    ) -> Tuple[float, Any, Any]:
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = jraph.get_graph_padding_mask(graph)  # [n_graphs,]
        loss, grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn(graph, model(params, graph)) * mask)
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

    last_cache_size = update_fn._cache_size()

    def interval_loader():
        i = 0
        while True:
            for graph in train_loader:
                yield graph
                i += 1
                if i >= steps_per_interval:
                    return

    for interval in itertools.count():
        yield interval, params, optimizer_state, ema_params

        # Train one interval
        p_bar = tqdm.tqdm(
            interval_loader(),
            desc=f"Train interval {interval}",
            total=steps_per_interval,
        )
        for graph in p_bar:
            num_updates += 1
            start_time = time.time()
            loss, params, optimizer_state, ema_params = update_fn(
                params, optimizer_state, ema_params, num_updates, graph
            )
            loss = float(loss)
            p_bar.set_postfix({"loss": f"{loss:7.3f}"})

            if last_cache_size != update_fn._cache_size():
                last_cache_size = update_fn._cache_size()

                #logging.info("Compiled function `update_fn` for args:")
                #logging.info(f"- n_node={graph.n_node} total={graph.n_node.sum()}")
                #logging.info(f"- n_edge={graph.n_edge} total={graph.n_edge.sum()}")
                #logging.info(f"Outout: loss= {loss:.3f}")
                #logging.info(
                #    f"Compilation time: {time.time() - start_time:.3f}s, cache size: {last_cache_size}"
                #)
                
                print("Compiled function `update_fn` for args:")
                #print(f"- n_node={graph.n_node} total={graph.n_node.sum()}")
                #print(f"- n_edge={graph.n_edge} total={graph.n_edge.sum()}")
                print(f"Outout: loss= {loss:.3f}")
                print(
                    f"Compilation time: {time.time() - start_time:.3f}s, cache size: {last_cache_size}"
                )


def evaluate(
    model: Callable,
    params: Any,
    loss_fn: Any,
    data_loader: GraphDataLoader,
    name: str = "Evaluation",
) -> Tuple[float, Dict[str, Any]]:
    total_loss = 0.0
    num_graphs = 0

    delta_es_list = []
    es_list = []

    delta_es_per_atom_list = []
    es_per_atom_list = []

    delta_fs_list = []
    fs_list = []

    delta_stress_list = []
    stress_list = []

    if hasattr(model, "_cache_size"):
        last_cache_size = model._cache_size()
    else:
        last_cache_size = None

    start_time = time.time()
    p_bar = tqdm.tqdm(data_loader, desc=name, total=data_loader.approx_length())
    for ref_graph in p_bar:
        output = model(params, ref_graph)
        pred_graph = ref_graph._replace(
            nodes=ref_graph.nodes._replace(forces=output["forces"]),
            globals=ref_graph.globals._replace(
                energy=output["energy"], stress=output["stress"]
            ),
        )

        if last_cache_size is not None and last_cache_size != model._cache_size():
            last_cache_size = model._cache_size()

            #logging.info("Compiled function `model` for args:")
            #logging.info(f"- n_node={ref_graph.n_node} total={ref_graph.n_node.sum()}")
            #logging.info(f"- n_edge={ref_graph.n_edge} total={ref_graph.n_edge.sum()}")
            #logging.info(f"cache size: {last_cache_size}")
            
            
            print("Compiled function `model` for args:")
            #print(f"- n_node={ref_graph.n_node} total={ref_graph.n_node.sum()}")
            #print(f"- n_edge={ref_graph.n_edge} total={ref_graph.n_edge.sum()}")
            print(f"cache size: {last_cache_size}")

        ref_graph = jraph.unpad_with_graphs(ref_graph)
        pred_graph = jraph.unpad_with_graphs(pred_graph)

        loss = jnp.sum(
            loss_fn(
                ref_graph,
                dict(
                    energy=pred_graph.globals.energy,
                    forces=pred_graph.nodes.forces,
                    stress=pred_graph.globals.stress,
                ),
            )
        )
        total_loss += float(loss)
        num_graphs += len(ref_graph.n_edge)
        p_bar.set_postfix({"n": num_graphs})

        if ref_graph.globals.energy is not None:
            delta_es_list.append(ref_graph.globals.energy - pred_graph.globals.energy)
            es_list.append(ref_graph.globals.energy)

            delta_es_per_atom_list.append(
                (ref_graph.globals.energy - pred_graph.globals.energy)
                / ref_graph.n_node
            )
            es_per_atom_list.append(ref_graph.globals.energy / ref_graph.n_node)

        if ref_graph.nodes.forces is not None:
            delta_fs_list.append(ref_graph.nodes.forces - pred_graph.nodes.forces)
            fs_list.append(ref_graph.nodes.forces)

        if ref_graph.globals.stress is not None:
            delta_stress_list.append(
                ref_graph.globals.stress - pred_graph.globals.stress
            )
            stress_list.append(ref_graph.globals.stress)

    if num_graphs == 0:
        #logging.warning(f"No graphs in data_loader ! Returning 0.0 for {name}")
        print(f"No graphs in data_loader ! Returning 0.0 for {name}")
        return 0.0, {}

    avg_loss = total_loss / num_graphs

    aux = {
        "loss": avg_loss,
        "time": time.time() - start_time,
        "mae_e": None,
        "rel_mae_e": None,
        "mae_e_per_atom": None,
        "rel_mae_e_per_atom": None,
        "rmse_e": None,
        "rel_rmse_e": None,
        "rmse_e_per_atom": None,
        "rel_rmse_e_per_atom": None,
        "q95_e": None,
        "mae_f": None,
        "rel_mae_f": None,
        "rmse_f": None,
        "rel_rmse_f": None,
        "q95_f": None,
        "mae_s": None,
        "rel_mae_s": None,
        "rmse_s": None,
        "rel_rmse_s": None,
        "q95_s": None,
    }

    if len(delta_es_list) > 0:
        delta_es = np.concatenate(delta_es_list, axis=0)
        delta_es_per_atom = np.concatenate(delta_es_per_atom_list, axis=0)
        es = np.concatenate(es_list, axis=0)
        es_per_atom = np.concatenate(es_per_atom_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                "mae_e": compute_mae(delta_es),
                "rel_mae_e": compute_rel_mae(delta_es, es),
                "mae_e_per_atom": compute_mae(delta_es_per_atom),
                "rel_mae_e_per_atom": compute_rel_mae(
                    delta_es_per_atom, es_per_atom
                ),
                # Root-mean-square error
                "rmse_e": compute_rmse(delta_es),
                "rel_rmse_e": compute_rel_rmse(delta_es, es),
                "rmse_e_per_atom": compute_rmse(delta_es_per_atom),
                "rel_rmse_e_per_atom": compute_rel_rmse(
                    delta_es_per_atom, es_per_atom
                ),
                # Q_95
                "q95_e": compute_q95(delta_es),
            }
        )
    if len(delta_fs_list) > 0:
        delta_fs = np.concatenate(delta_fs_list, axis=0)
        fs = np.concatenate(fs_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                "mae_f": compute_mae(delta_fs),
                "rel_mae_f": compute_rel_mae(delta_fs, fs),
                # Root-mean-square error
                "rmse_f": compute_rmse(delta_fs),
                "rel_rmse_f": compute_rel_rmse(delta_fs, fs),
                # Q_95
                "q95_f": compute_q95(delta_fs),
            }
        )
    if len(delta_stress_list) > 0:
        delta_stress = np.concatenate(delta_stress_list, axis=0)
        stress = np.concatenate(stress_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                "mae_s": compute_mae(delta_stress),
                "rel_mae_s": compute_rel_mae(delta_stress, stress),
                # Root-mean-square error
                "rmse_s": compute_rmse(delta_stress),
                "rel_rmse_s": compute_rel_rmse(delta_stress, stress),
                # Q_95
                "q95_s": compute_q95(delta_stress),
            }
        )

    return avg_loss, aux




def energy_force_train(
    model,
    params,
    optimizer_state,
    train_loader,
    valid_loader,
    test_loader,
    gradient_transform,
    loss_fn,
    max_num_intervals: int,
    steps_per_interval: int,
    save_dir_name: str,
    *,
    patience: Optional[int] = None,
    eval_train: bool = True,
    eval_valid: bool = True,
    eval_test: bool = False,
    log_errors: str = "PerAtomMAE",
    **kwargs,
):
    lowest_loss = np.inf
    patience_counter = 0
    #loss_fn = loss_fn
    start_time = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []

    for interval, params, optimizer_state, ema_params in train(
        model=model,
        params=params,
        loss_fn=loss_fn,
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=steps_per_interval,
        **kwargs,
    ):
        total_time_per_interval += [time.perf_counter() - start_time]
        start_time = time.perf_counter()

        try:
            import profile_nn_jax
        except ImportError:
            pass
        else:
            profile_nn_jax.restart_timer()

        last_interval = interval == max_num_intervals
        
        with open(f"{save_dir_name}/params.pkl", "wb") as f:
            pickle.dump(params, f)

        with open(f"{save_dir_name}/ema_params.pkl", "wb") as f:
            pickle.dump(ema_params, f)

        def eval_and_print(loader, mode: str):
            loss_, metrics_ = evaluate(
                model=model,
                params=ema_params,
                loss_fn=loss_fn,
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
                    return f"{100 * v:.1f}%"
                if "_e" in x:
                    return f"{1e3 * v:.1f} meV"
                if "_f" in x:
                    return f"{1e3 * v:.1f} meV/Å"
                if "_s" in x:
                    return f"{1e3 * v:.1f} meV/Å³"
                raise NotImplementedError

            #logging.info(
            #    f"Interval {interval}: {mode}: "
            #    f"loss={loss_:.4f}, "
            #    f"{error_e}={_(error_e)}, "
            #    f"{error_f}={_(error_f)}, "
            #    f"{error_s}={_(error_s)}"
            #)
            
            print(
                f"Interval {interval}: {mode}: "
                f"loss={loss_:.4f}, "
                f"{error_e}={_(error_e)}, "
                f"{error_f}={_(error_f)}, "
                f"{error_s}={_(error_s)}"
            )
            
            
            return loss_, metrics_

        if eval_train or last_interval:
            #if not isinstance(eval_train, bool):
            #    eval_and_print(train_loader.subset(eval_train), "eval_train")
            #else:
            #    eval_and_print(train_loader, "eval_train")
            loss_, metrics_ = eval_and_print(train_loader, "eval_train")
            #print('check metric', metrics_)
            
            with open(f"{save_dir_name}/metrics_train.json", "a") as f:
                yaml.dump([{"interval": interval, "mode" : "eval_train" , **metrics_}], f)
                
        if eval_valid and (valid_loader is not None): # and len(valid_loader) > 0:
            loss_, metrics_  = eval_and_print(valid_loader, "eval_valid")
            
            with open(f"{save_dir_name}/metrics_valid.json", "a") as f:
                yaml.dump([{"interval": interval, "mode" : "eval_valid", **metrics_}], f)
            

        if (
            (eval_test or last_interval)
            and test_loader is not None
            and len(test_loader) > 0
        ):
            loss_, metrics_ = eval_and_print(test_loader, "eval_test")
            
            
            with open(f"{save_dir_name}/metrics_valid.json", "a") as f:
                yaml.dump([{"interval": interval, **metrics_}], f)

        #if valid_loader is not None and len(valid_loader) > 0:
        #    loss_ = eval_and_print(valid_loader, "eval_valid")

        #    if loss_ >= lowest_loss:
        #        patience_counter += 1
        #        if patience is not None and patience_counter >= patience:
        #            logging.info(
        #                f"Stopping optimization after {patience_counter} intervals without improvement"
        #            )
        #            break
        #    else:
        #        lowest_loss = loss_
        #        patience_counter = 0

        eval_time_per_interval += [time.perf_counter() - start_time]
        avg_time_per_interval = np.mean(total_time_per_interval[-3:])
        avg_eval_time_per_interval = np.mean(eval_time_per_interval[-3:])

        #logging.info(
        #    f"Interval {interval}: Time per interval: {avg_time_per_interval:.1f}s, "
        #    f"among which {avg_eval_time_per_interval:.1f}s for evaluation."
        #)

        if last_interval:
            break

    #logging.info("Training complete")
    print("Training complete")
    return interval, params, ema_params


def load_metrics(
    fname:str,
):
    with open(fname,'r') as f:
        data_ = yaml.load(f,Loader=yaml.SafeLoader)
        
    if len(data_) == 0:
        return {}
        
    metrics_data = {}
    for key_ in data_[0].keys():
        metrics_data[key_] = [metrics_[key_] for metrics_ in data_]

    return metrics_data


