import jax.numpy as jnp
import jraph
import optax

from .utils import (
    sum_nodes_of_the_same_graph, 
    _safe_divide,
)


def mean_squared_error_energy(graph, energy_pred) -> jnp.ndarray:
    energy_ref = graph.globals.energy  # [n_graphs, ]
    return graph.globals.weight * jnp.square(
        _safe_divide(energy_ref - energy_pred, graph.n_node)
    )  # [n_graphs, ]


def mean_squared_error_forces(graph, forces_pred) -> jnp.ndarray:
    forces_ref = graph.nodes.forces  # [n_nodes, 3]
    return graph.globals.weight * _safe_divide(
        sum_nodes_of_the_same_graph(
            graph, jnp.mean(jnp.square(forces_ref - forces_pred), axis=1)
        ),
        graph.n_node,
    )  # [n_graphs, ]


def mean_squared_error_stress(graph, stress_pred) -> jnp.ndarray:
    stress_ref = graph.globals.stress  # [n_graphs, 3, 3]
    return graph.globals.weight * jnp.mean(
        jnp.square(stress_ref - stress_pred), axis=(1, 2)
    )  # [n_graphs, ]


class WeightedEnergyFrocesStressLoss:
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = 0

        if self.energy_weight > 0.0:
            energy = predictions["energy"]
            loss += self.energy_weight * mean_squared_error_energy(graph, energy)

        if self.forces_weight > 0.0:
            forces = predictions["forces"]
            loss += self.forces_weight * mean_squared_error_forces(graph, forces)

        if self.stress_weight > 0.0:
            stress = predictions["stress"]
            loss += self.stress_weight * mean_squared_error_stress(graph, stress)

        return loss  # [n_graphs, ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"stress_weight={self.stress_weight:.3f})"
        )

    
def mean_squared_error_energy_Huber(graph, energy_pred,huber_delta) -> jnp.ndarray:
    energy_ref = graph.globals.energy  # [n_graphs, ]
    return graph.globals.weight * _safe_divide( optax.huber_loss(energy_pred, energy_ref, huber_delta), graph.n_node)
    
    
def mean_squared_error_forces_Huber(graph, forces_pred, huber_delta) -> jnp.ndarray:
    forces_ref = graph.nodes.forces  # [n_nodes, 3]

    return graph.globals.weight * _safe_divide(
        sum_nodes_of_the_same_graph(
            graph, jnp.mean(optax.huber_loss(forces_pred , forces_ref,huber_delta), axis=1)
        ),
        graph.n_node,
    )  # [n_graphs, ]

def mean_squared_error_stress_Huber(graph, stress_pred, huber_delta) -> jnp.ndarray:
    stress_ref = graph.globals.stress  # [n_graphs, 3, 3]
    return graph.globals.weight * jnp.mean(
        optax.huber_loss(stress_pred , stress_ref,huber_delta), axis=(1, 2)
    )  # [n_graphs, ]

    
class WeightedEnergyFrocesStressLoss_Huber:
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=1.0) -> None:
        super().__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.huber_delta = huber_delta

        #self.huber_loss = huber_delta

    def __call__(self, graph: jraph.GraphsTuple, predictions) -> jnp.ndarray:
        loss = 0

        if self.energy_weight > 0.0:
            energy = predictions["energy"]
            loss += self.energy_weight * mean_squared_error_energy_Huber(graph, energy, self.huber_delta)

        if self.forces_weight > 0.0:
            forces = predictions["forces"]
            loss += self.forces_weight * mean_squared_error_forces_Huber(graph, forces, self.huber_delta)

        if self.stress_weight > 0.0:
            stress = predictions["stress"]
            loss += self.stress_weight * mean_squared_error_stress_Huber(graph, stress, self.huber_delta)

        return loss  # [n_graphs, ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"stress_weight={self.stress_weight:.3f})"
        )

def abs2(z: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(z.real**2 + z.imag**2)

class HessianLoss:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self,graph: jraph.GraphsTuple, predictions: jnp.ndarray) -> jnp.ndarray:
        # v0
        #return jnp.square(predictions - graph.globals.hessian.squeeze())
        return abs2(predictions - graph.globals.hessian.squeeze())
        
    def __repr__(self):
        return (
            "hessian loss fn"
        )
    
class crystalHessianLoss:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self,graph: jraph.GraphsTuple, predictions: jnp.ndarray) -> jnp.ndarray:
        # v0
        return abs2(predictions- graph.globals.dynmat)
        
    def __repr__(self):
        return (
            "crystal hessian loss fn"
        )
    
    