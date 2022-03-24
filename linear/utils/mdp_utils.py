# =============================================================================
# Utility functions for solving MDPs
#
# Author: XX
# =============================================================================

import gym
import numpy as np


def compute_rmse(vec_a, vec_b) -> float:
    """
    Compute the root mean square error (RMSE) between two vectors
    :param vec_a: np array of shape (N, )
    :param vec_b: np array of shape (N, )
    :return: scalar
    """
    sq_err = (vec_a - vec_b) ** 2
    return np.sqrt(np.mean(sq_err))


def solve_value_fn(env: gym.Env, gamma: float) -> np.ndarray:
    """
    Solve the value function for each state in an environment
    :param env: gym environment
    :param gamma: discount factor
    :return:
    """
    # Transition matrix
    n_states = env.get_num_states()
    P_trans = env.get_transition_matrix()

    # Reward function
    R_fn = env.get_reward_function()

    # Solve and return
    c_mat = (np.identity(n_states) - (gamma * P_trans))
    v_fn = np.linalg.inv(c_mat) @ R_fn

    return v_fn


def solve_successor_feature(env: gym.Env, gamma: float) -> np.ndarray:
    """
    Analytically solve for the successor feature
    :param env: gym environment
    :param gamma: discount factor
    :return: (N, d) numpy matrix of successor feature for each state
    """
    # Transition matrix
    n_states = env.get_num_states()
    P_trans = env.get_transition_matrix()

    # Feature matrix
    Phi_mat = env.get_feature_matrix()

    # Solve and return
    c_mat = (np.identity(n_states) - (gamma * P_trans))
    sf_mat = np.linalg.inv(c_mat) @ Phi_mat
    return sf_mat


def solve_linear_sf_param(env: gym.Env, gamma: float) -> np.ndarray:
    """
    Solve for the linear successor feature parameter matrix. Given an
    environment, extract the transition and feature matrices. Solve the
    best linear approximation to the perfect successor feature.

    :param env: gym environment
    :param gamma: float discount factor
    :return: np.array of (d, d) successor matrix
    """
    phiMat = env.get_feature_matrix()  # (N, d) feature mat
    transMat = env.get_transition_matrix()  # (N, N) trans mat
    p_n_states = np.shape(transMat)[0]  # N

    # Project and solve
    cMat = np.identity(p_n_states) - (gamma * transMat)
    proj_cMat = phiMat.T @ cMat @ phiMat
    Z = np.linalg.inv(proj_cMat) @ (phiMat.T @ phiMat)

    return Z


def solve_linear_reward_param(env: gym.Env) -> np.ndarray:
    """
    Solve for the linear (one-step) reward parameter vector.
    :param env:  gym environment
    :return: (d, ) reward function parameters
    """
    phiMat = env.get_feature_matrix()  # (N, d) feature mat
    rewVec = env.get_reward_function()  # (N, 1) reward vec

    # Project and solve
    solMat = np.linalg.inv((phiMat.T @ phiMat))
    Wr = solMat @ phiMat.T @ rewVec

    return Wr


def evaluate_value_rmse(env: gym.Env, agent, true_v_fn) -> float:
    """
    Compute the RMSE for the value function of a given agent
    and environment

    :return: scalar RMSE
    """

    phiMat = env.get_feature_matrix()  # (N, d) feature mat
    n_states = np.shape(phiMat)[0]
    esti_v_fn = np.empty(n_states)

    for s_n in range(n_states):
        # Get state features
        s_phi = phiMat[s_n, :]

        esti_v_fn[s_n] = agent.compute_Q_value(s_phi, 0)

    return compute_rmse(esti_v_fn, true_v_fn)


def evaluate_sf_ret_rmse(env, agent, true_v_fn) -> float:
    """
    Compute RMSE for the lambda successor return, if possible
    :return: scalar RMSE
    """
    if not hasattr(agent, 'compute_successor_return'):
        return None

    phiMat = env.get_feature_matrix()  # (N, d) feature mat
    n_states = np.shape(phiMat)[0]
    esti_v_fn = np.empty(n_states)

    for s_n in range(n_states):
        s_phi = phiMat[s_n, :]  # state features
        esti_v_fn[s_n] = agent.compute_successor_return(
            s_phi, 0
        )  # compute value
    return compute_rmse(esti_v_fn, true_v_fn)


def evaluate_sf_mat_rmse(env, agent, true_sf_mat) -> float:
    """
    Compute the RMSE for the successor feature matrix
    :param env:
    :param agent:
    :param true_sf_mat:  (N, d) true successor feature matrix
    :return:
    """
    # NOTE: assumes only single action, and assumes parameter name is Ws
    sf_param = agent.Ws[0]  # (d, d)
    phiMat = env.get_feature_matrix()  # (N, d) feature mat

    esti_sf_mat = phiMat @ sf_param

    return compute_rmse(esti_sf_mat, true_sf_mat)


def evaluate_reward_rmse(env, agent, true_rew_vec) -> float:
    """
    Compute the RMSE for the (instantenous) reward estimates
    :param env:
    :param agent:
    :return:
    """

    # NOTE: assumes parameter is available
    rew_param = agent.Wr  # (d,) vec
    phiMat = env.get_feature_matrix()  # (N, d) feature mat

    esti_rew_vec = phiMat @ rew_param
    return compute_rmse(esti_rew_vec, true_rew_vec)


if __name__ == "__main__":
    print('hello world')
