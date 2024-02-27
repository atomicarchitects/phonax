import numpy as np


def _message_passing(
    ext_senders_in_cell0,
    ext_receivers_in_cell0,
    ext_senders_unit_shifts_from_cell0,
    ext_receivers_unit_shifts_from_cell0,
    senders,
    receivers,
    senders_unit_shifts,
):
    x = np.unique(
        np.concatenate(
            [
                np.concatenate(
                    [
                        ext_senders_in_cell0[:, None],
                        ext_senders_unit_shifts_from_cell0,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        ext_receivers_in_cell0[:, None],
                        ext_receivers_unit_shifts_from_cell0,
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        ),
        axis=0,
    )
    node_index_in_cell0, node_shifts_from_cell0 = x[:, 0], x[:, 1:4]

    new_edges = []

    for i, us in zip(node_index_in_cell0, node_shifts_from_cell0):
        mask = receivers == i
        s = senders[mask]
        r = receivers[mask]
        s_us = us + senders_unit_shifts[mask]
        r_us = us + np.zeros_like(s_us)

        new_edges += [np.concatenate([s[:, None], r[:, None], s_us, r_us], axis=1)]

    new_edges = np.concatenate(new_edges, axis=0)
    new_edges = np.unique(new_edges, axis=0)

    ext_senders_in_cell0 = new_edges[:, 0]
    ext_receivers_in_cell0 = new_edges[:, 1]
    ext_senders_unit_shifts_from_cell0 = new_edges[:, 2:5]
    ext_receivers_unit_shifts_from_cell0 = new_edges[:, 5:8]

    return (
        ext_senders_in_cell0,
        ext_receivers_in_cell0,
        ext_senders_unit_shifts_from_cell0,
        ext_receivers_unit_shifts_from_cell0,
    )


def into_concrete_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    ext_senders_in_cell0: np.ndarray,
    ext_receivers_in_cell0: np.ndarray,
    ext_senders_unit_shifts_from_cell0: np.ndarray,
    ext_receivers_unit_shifts_from_cell0: np.ndarray,
):
    """Convert the graph into a concrete graph.

    Args:
        positions (np.ndarray): Positions of the original nodes. Shape ``(num_nodes, 3)``.
        cell (np.ndarray): The cell matrix. Shape ``(3, 3)``.
        senders (np.ndarray): The senders of the edges. Pointing to the original nodes. Shape ``(num_edges,)``.
        receivers (np.ndarray): The receivers of the edges. Pointing to the original nodes. Shape ``(num_edges,)``.
        senders_unit_shifts (np.ndarray): The unit shifts of the senders. Shape ``(num_edges, 3)``.
        receivers_unit_shifts (np.ndarray): The unit shifts of the receivers. Shape ``(num_edges, 3)``.

    Returns:
        positions (np.ndarray): The positions of the nodes. Shape ``(num_new_nodes, 3)``.
            The first ``num_nodes`` nodes are the same as the input but not necessarily in the same order.
        original_node_indices (np.ndarray): The indices of the original nodes. Shape ``(num_new_nodes,)``.
        original_unit_shifts (np.ndarray): The unit shifts with respect to the original nodes. Shape ``(num_new_nodes, 3)``.
        senders (np.ndarray): The senders of the edges. Pointing to the new nodes. Shape ``(num_new_edges,)``.
        receivers (np.ndarray): The receivers of the edges. Pointing to the new nodes. Shape ``(num_new_edges,)``.
    """
    x, i = np.unique(
        np.concatenate(  # The list of all nodes, created by all the edges
            [
                np.concatenate(
                    [
                        np.sum(
                            ext_senders_unit_shifts_from_cell0**2,
                            axis=1,
                            keepdims=True,
                        ),
                        ext_senders_unit_shifts_from_cell0,
                        ext_senders_in_cell0[:, None],  # index in 0th cell
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        np.sum(
                            ext_receivers_unit_shifts_from_cell0**2,
                            axis=1,
                            keepdims=True,
                        ),
                        ext_receivers_unit_shifts_from_cell0,
                        ext_receivers_in_cell0[:, None],
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        ),
        axis=0,
        return_inverse=True,
    )
    ext_node_unit_shifts_from_cell0, ext_node_index_in_cell0 = x[:, 1:4], x[:, 4]

    ext_node_positions = (
        positions[ext_node_index_in_cell0] + ext_node_unit_shifts_from_cell0 @ cell
    )

    ext_senders = i[: len(ext_senders_in_cell0)]
    ext_receivers = i[len(ext_senders_in_cell0) :]

    # Sort for fun (not necessary)
    j = np.lexsort((ext_senders, ext_receivers))
    ext_senders = ext_senders[j]
    ext_receivers = ext_receivers[j]

    return (
        ext_node_positions,
        ext_node_index_in_cell0,
        ext_node_unit_shifts_from_cell0,
        ext_senders,
        ext_receivers,
    )


def pad_periodic_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    senders_unit_shifts: np.ndarray,
    num_message_passing: int,
):
    """Pad a graph with periodic boundary conditions. Return an extended graph.

    Args:
        positions (np.ndarray): Positions of the nodes in the 0th cell. Shape ``(num_nodes, 3)``.
        cell (np.ndarray): The cell matrix. Shape ``(3, 3)``.
        senders (np.ndarray): The senders of the edges. Pointing to the original nodes. Shape ``(num_edges,)``.
        receivers (np.ndarray): The receivers of the edges. Pointing to the original nodes. Shape ``(num_edges,)``.
        senders_unit_shifts (np.ndarray): The unit shifts of the senders. Shape ``(num_edges, 3)``.
            position of sender = positions[sender] + senders_unit_shifts @ cell
        num_message_passing (int): The number of message passing steps.

    Returns:
        ext_node_positions (np.ndarray): The positions of the new nodes. Shape ``(num_new_nodes, 3)``.
            The first ``num_nodes`` nodes are the same as the input but not necessarily in the same order.
        ext_node_index_in_cell0 (np.ndarray): The corresponding indices in 0th cell. Shape ``(num_new_nodes,)``.
        ext_node_unit_shifts_from_cell0 (np.ndarray): The unit shifts with respect to 0th cell. Shape ``(num_new_nodes, 3)``.
        ext_senders (np.ndarray): The senders of the extended edges. Pointing to the new nodes. Shape ``(num_new_edges,)``.
        ext_receivers (np.ndarray): The receivers of the extended edges. Pointing to the new nodes. Shape ``(num_new_edges,)``.
    """
    ext_senders_in_cell0 = senders
    ext_receivers_in_cell0 = receivers

    ext_senders_unit_shifts_from_cell0 = senders_unit_shifts
    ext_receivers_unit_shifts_from_cell0 = np.zeros_like(senders_unit_shifts)

    for _ in range(num_message_passing - 1):
        (
            ext_senders_in_cell0,
            ext_receivers_in_cell0,
            ext_senders_unit_shifts_from_cell0,
            ext_receivers_unit_shifts_from_cell0,
        ) = _message_passing(
            ext_senders_in_cell0,
            ext_receivers_in_cell0,
            ext_senders_unit_shifts_from_cell0,
            ext_receivers_unit_shifts_from_cell0,
            senders,
            receivers,
            senders_unit_shifts,
        )

    return into_concrete_graph(
        positions,
        cell,
        ext_senders_in_cell0,
        ext_receivers_in_cell0,
        ext_senders_unit_shifts_from_cell0,
        ext_receivers_unit_shifts_from_cell0,
    )


def plot_extended_graph(
    ext_node_positions: np.ndarray,
    ext_node_index_in_cell0: np.ndarray,
    ext_node_unit_shifts_from_cell0: np.ndarray,
    ext_senders: np.ndarray,
    ext_receivers: np.ndarray,
    x_axis: int = 0,
    y_axis: int = 1,
):
    """Plot the extended graph."""
    import matplotlib.pyplot as plt

    positions_sender = ext_node_positions[ext_senders]
    positions_receiver = ext_node_positions[ext_receivers]
    mask_in_cell0 = np.all(ext_node_unit_shifts_from_cell0 == 0, axis=1)

    plt.scatter(
        ext_node_positions[mask_in_cell0, x_axis],
        ext_node_positions[mask_in_cell0, y_axis],
        c="black",
        zorder=9,
        s=10,
    )
    plt.scatter(
        ext_node_positions[:, x_axis],
        ext_node_positions[:, y_axis],
        c=ext_node_index_in_cell0,
        zorder=10,
        cmap="tab10",
        s=5,
    )
    plt.axis("equal")

    for i in range(len(ext_senders)):
        x = positions_receiver[i, x_axis] - positions_sender[i, x_axis]
        y = positions_receiver[i, y_axis] - positions_sender[i, y_axis]
        length = np.sqrt(x**2 + y**2)

        if length > 0.2:
            x = x / length * (length - 0.2)
            y = y / length * (length - 0.2)
            plt.arrow(
                positions_sender[i, x_axis],
                positions_sender[i, y_axis],
                x,
                y,
                head_width=0.05,
                head_length=0.08,
                color="black",
                alpha=0.5,
            )

    plt.xticks([])
    plt.yticks([])
