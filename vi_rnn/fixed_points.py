# SCYFI code adapted from:
#  (Eisenmann et al. 2023, GNU General Public Licence)
# https://github.com/DurstewitzLab/SCYFI

import numpy as np


def get_cycle_point_candidate(A, W1, W2, h1, h2, D_list, order, tol=1e-8):
    """
    Get the candidate for a cycle point by solving the cycle equation
    (finds 'virtual' fixed points of order-times iterated system)

    Uses np.linalg.solve instead of explicit inverse for stability.
    Checks near-singularity of the matrix.

    Args:
        A, W1, W2 : RNN matrices
        h1, h2 : biases
        D_list : list of ReLU masks
        order : cycle order
        tol : threshold for detecting near-singularity

    Returns:
        z_candidate (np.ndarray) or None if unsolvable
    """
    z_factor, h1_factor, h2_factor = get_factors(A, W1, W2, D_list, order)

    M = np.eye(A.shape[0]) - z_factor
    b = h1_factor @ h1 + h2_factor @ h2

    if np.linalg.cond(M) > 1 / tol:
        # Matrix is too ill-conditioned or singular
        return None

    try:
        z_candidate = np.linalg.solve(M, b)
        return z_candidate
    except np.linalg.LinAlgError:
        return None


def get_factors(A, W1, W2, D_list, order):
    """
    Build the matrix factors for the k-cycle fixed-point equation.

    Args:
        A (np.ndarray; dim_z x dim_z): latent decay matrix
        W1 (np.ndarray; dim_z x N): right connectivity
        W2 (np.ndarray; N x dim_z): left connectivity
        D_list (np.ndarray; N x order): ReLU activation masks per cycle step
        order (int): cycle order

    Returns:
        factor_z (np.ndarray): cumulative transition factor on z
        factor_h1 (np.ndarray): cumulative factor on h1
        factor_h2 (np.ndarray): cumulative factor on h2
    """
    hidden_dim = W2.shape[0]
    latent_dim = W1.shape[0]
    factor_z = np.eye(A.shape[0])
    factor_h1 = np.eye(A.shape[0])
    factor_h2 = (W1 * D_list[:, 0]).dot(np.eye(hidden_dim))
    for i in range(order - 1):
        factor_z = (A + (W1 * D_list[:, i]).dot(W2)).dot(factor_z)
        factor_h1 = (A + (W1 * D_list[:, i + 1]).dot(W2)).dot(factor_h1) + np.eye(
            A.shape[0]
        )
        factor_h2 = (A + (W1 * D_list[:, i + 1]).dot(W2)).dot(factor_h2) + (
            W1 * D_list[:, i + 1]
        )
    factor_z = (A + (W1 * D_list[:, order - 1]).dot(W2)).dot(factor_z)
    return factor_z, factor_h1, factor_h2


def get_latent_time_series(time_steps, A, W1, W2, h1, h2, dz, z_0=None):
    """
    Roll out a piecewise-linear RNN trajectory for ``time_steps`` steps.

    Args:
        time_steps (int): number of steps to simulate
        A, W1, W2, h1, h2: PLRNN parameters
        dz (int): latent dimensionality (used only when ``z_0`` is None)
        z_0 (np.ndarray, optional): initial latent state

    Returns:
        trajectory (list): latent states, length ``time_steps``
    """
    if z_0 is None:
        z = np.random.randn(dz)
    else:
        z = z_0
    trajectory = [z]

    for t in range(1, time_steps):
        z = latent_step(z, A, W1, W2, h1, h2)
        trajectory.append(z)
    return trajectory


def latent_step(z, A, W1, W2, h1, h2):
    """One discrete step of a piecewise-linear RNN dynamics map.

    Args:
        z: Latent state vector.
        A, W1, W2, h1, h2: PLRNN parameters (decay, weights, biases).

    Returns:
        z_next: Updated latent state after one step.
    """
    return A.dot(z) + W1.dot(np.maximum(W2.dot(z) + h2, 0)) + h1


def check_fixed_points_one_step(A, W1, W2, h1, h2, fixed_points):
    """Step each fixed point once and report mean max deviation from invariance.

    Args:
        A, W1, W2, h1, h2: PLRNN parameters passed to ``latent_step``.
        fixed_points: ``(n_fps, dim_z)`` array, or SCYFI trajectories
            ``(n_fps, order, dim_z)`` / list of trajectory lists.

    Returns:
        Mean over fixed points of ``max(|z' - z|)`` after one dynamics step.
    """
    fps = np.asarray(fixed_points, dtype=float)
    if fps.ndim == 3:
        fps = fps[:, 0]
    elif fps.ndim == 1:
        fps = fps[None, :]

    if fps.size == 0:
        print("Fixed-point one-step check: no fixed points (mean max deviation = nan)")
        return np.nan

    max_devs = []
    for z in fps:
        z_next = latent_step(z, A, W1, W2, h1, h2)
        max_devs.append(np.max(np.abs(z_next - z)))

    mean_max_dev = float(np.mean(max_devs))
    print(
        f"Fixed-point one-step check: mean max deviation = {mean_max_dev:.3e} "
        f"(n={len(fps)})"
    )
    return mean_max_dev


def get_eigvals(A, W1, W2, D_list, order):
    """
    Compute eigenvalues of the k-step Jacobian for discrete-time stability.

    Args:
        A (np.ndarray; dim_z x dim_z): latent decay matrix
        W1 (np.ndarray; dim_z x N): right connectivity
        W2 (np.ndarray; N x dim_z): left connectivity
        D_list (np.ndarray; N x order): ReLU activation masks per cycle step
        order (int): cycle order

    Returns:
        eigvals (np.ndarray): eigenvalues of the composed Jacobian
    """
    # Initialize the cumulative Jacobian as an Identity matrix
    # for a k-cycle, J_total = J_k * J_{k-1} * ... * J_1
    J_total = np.eye(A.shape[0])

    # Ensure A is a 2D matrix if it was passed as a diagonal vector
    A_mat = np.diag(A) if A.ndim == 1 else A

    for i in range(order):
        #  Get the ReLU mask for this step in the cycle
        # D_list[:, i] is the binary vector of which neurons are active
        # W1 * D_list[:, i] zeroes out the columns of W1 for inactive neurons
        W1_active = W1 * D_list[:, i]

        # Calculate the Jacobian for this specific linear region
        # J = A + W1 * D * W2
        J_i = A_mat + W1_active.dot(W2)

        # Accumulate the product
        J_total = J_i.dot(J_total)

    return np.linalg.eigvals(J_total)


def scy_fi(
    A,
    W1,
    W2,
    h1,
    h2,
    order,
    found_lower_orders,
    inner_loop_iterations=100,
    round_dec=2,
    n_inverses_max=1000,
    initial_states=None,
):
    """
    Heuristic SCYFI search for fixed points and k-cycles.

    Args:
        A (np.ndarray; dim_z x dim_z): latent decay matrix
        W1 (np.ndarray; dim_z x N): right connectivity
        W2 (np.ndarray; N x dim_z): left connectivity
        h1 (np.ndarray; dim_z): latent bias
        h2 (np.ndarray; N): hidden bias
        order (int): cycle order to search
        found_lower_orders (list): previously found lower-order cycles
        inner_loop_iterations (int): inner refinement iterations per initial state
        round_dec (int): decimal places for duplicate detection
        n_inverses_max (int): maximum number of linear solves
        initial_states (np.ndarray): ReLU pattern seeds, shape ``(N, N)`` or ``(N,)``

    Returns:
        cycles_found (list): trajectories of found cycles
        eigvals (list): stability eigenvalues per found cycle
        n_inverses (int): number of linear solves performed
    """
    hidden_dim = h2.shape[0]
    latent_dim = h1.shape[0]
    cycles_found = []
    D_list_cycles = []
    eigvals = []
    n_inverses = 0
    if initial_states is None:
        print("No initial states provided.")
        return [], [], 0
    i = -1
    while i < len(initial_states) - 1 and n_inverses < n_inverses_max:
        print(
            f"Outer loop iteration {i+1}, found {len(cycles_found)} cycles, n_inverses={n_inverses}",
            end="\r",
        )
        i += 1
        # If initial_states[i] is a single column, we tile it to match the cycle order.
        current_init = initial_states[i]
        if current_init.ndim == 1:
            relu_matrix_list = np.tile(current_init[:, None], (1, order))
        else:
            relu_matrix_list = current_init

        c = 0
        while c < inner_loop_iterations and n_inverses < n_inverses_max:
            c += 1
            z_candidate = get_cycle_point_candidate(
                A, W1, W2, h1, h2, relu_matrix_list, order
            )
            n_inverses += 1

            if z_candidate is not None:
                trajectory = get_latent_time_series(
                    order, A, W1, W2, h1, h2, latent_dim, z_0=z_candidate
                )

                # Calculate the new ReLU pattern
                trajectory_relu_matrix_list = np.empty((hidden_dim, order))
                for j in range(order):
                    trajectory_relu_matrix_list[:, j] = (W2.dot(trajectory[j]) + h2) > 0

                # Check if this matches our search pattern
                diff = np.sum(np.abs(trajectory_relu_matrix_list - relu_matrix_list))

                if diff == 0:
                    # SUCCESS: We found a consistent cycle
                    # Now check if it's a lower order or duplicate
                    is_valid = True
                    if found_lower_orders:
                        # Check against known lower order cycles
                        for low_cycle in found_lower_orders:
                            if np.allclose(
                                trajectory[0], low_cycle, atol=10**-round_dec
                            ):
                                is_valid = False
                                break

                    if is_valid:
                        current_pattern = trajectory_relu_matrix_list.copy()
                        is_duplicate = any(
                            any(
                                np.array_equal(
                                    np.roll(current_pattern, s, axis=1), existing
                                )
                                for s in range(order)
                            )
                            for existing in D_list_cycles
                        )

                        if not is_duplicate:
                            cycles_found.append(trajectory.copy())
                            D_list_cycles.append(current_pattern)
                            eigvals.append(
                                get_eigvals(A, W1, W2, relu_matrix_list, order)
                            )

                    # Since we found a consistent point (valid or not),
                    # we break this inner loop to try the next initial state.
                    break

                else:
                    # The pattern changed, so we update and continue the inner loop
                    relu_matrix_list = trajectory_relu_matrix_list.copy()
            else:
                # Singularity/Failure
                break
    return cycles_found, eigvals, n_inverses


def run_scify(
    A,
    W1,
    W2,
    h1,
    h2,
    order=1,
    inner_loop_iterations=100,
    round_dec=4,
    n_inverses_max=1000,
    initial_states=None,
):
    """
    Run SCYFI fixed-point search for all cycle orders up to ``order``.

    Args:
        A (np.ndarray; dim_z x dim_z): latent decay matrix
        W1 (np.ndarray; dim_z x N): right connectivity
        W2 (np.ndarray; N x dim_z): left connectivity
        h1 (np.ndarray; dim_z): latent bias
        h2 (np.ndarray; N): hidden bias
        order (int): maximum cycle order to search
        inner_loop_iterations (int): inner refinement iterations per initial state
        round_dec (int): decimal places for duplicate detection
        n_inverses_max (int): maximum number of linear solves per order
        initial_states (np.ndarray): ReLU pattern seeds

    Returns:
        found_lower_orders (list): found cycles per order (list of lists)
        found_eigvals (list): eigenvalues per order (list of lists)
        n_inverses (list): linear solve counts per order
    """

    found_lower_orders = []
    found_eigvals = []
    n_inverses = []
    for i in range(1, order + 1):
        cycles_found, eigvals, n_inverses_ = scy_fi(
            A,
            W1,
            W2,
            h1,
            h2,
            i,
            found_lower_orders,
            inner_loop_iterations=inner_loop_iterations,
            round_dec=round_dec,
            n_inverses_max=n_inverses_max,
            initial_states=initial_states,
        )
        found_lower_orders.append(cycles_found)
        found_eigvals.append(eigvals)
        n_inverses.append(n_inverses_)
    return found_lower_orders, found_eigvals, n_inverses
