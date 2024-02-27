import logging
import pickle
import jax





def reload(params, path=None):
    if path is not None:
        logging.info(f"Reloading parameters from '{path}'")
        with open(path, "rb") as f:
            _ = pickle.load(f)
            new_params = pickle.load(f)

        # check compatibility
        if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(
            new_params
        ):
            logging.warning(
                f"Parameters from '{path}' are not compatible with current model"
            )

        return new_params
    return params


