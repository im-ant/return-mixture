#!/bin/bash
# ============================================================================
# Example script
# ============================================================================

# ===========================
# Experimental set-up

# (Load package and environment example)
# module load python/3.7
# module load python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0
# source $HOME/venvs/rl/bin/activate

# Run job sweep
base_dir='./output/year-month-day/${now:%H-%M-%S}'
add_name="_s$SEEDS"
sweep_parent_dir=$base_dir$add_name

python -u train_nonlinear.py \
  hydra/job_logging=default \
  runner=batched_offline \
  runner.n_steps=5e6 \
  runner.kwargs.log_interval_episodes=10 \
  runner.kwargs.store_checkpoint=False \
  runner.kwargs.replay_start_buffer_size=5000 \
  runner.buffer_kwargs.buffer_size=100000 \
  env.kwargs.env_name='breakout' \
  algo=lsf_dqn \
  algo.kwargs.sf_lambda=0.5 \
  algo.kwargs.start_epsilon=1.0 \
  algo.kwargs.end_epsilon=0.05 \
  algo.kwargs.initial_epsilon_length=5000 \
  algo.kwargs.epsilon_anneal_length=100000 \
  algo.kwargs.use_target_net=True \
  algo.kwargs.policy_updates_per_target_update=1000 \
  algo.kwargs.optim_kwargs.lr=0.00025 \
  algo.kwargs.sf_optim_kwargs.lr=0.005 \
  algo.kwargs.reward_optim_kwargs.lr=0.005 \
  model=lsf_q_network \
  model.cls_string='LQNet_sharePsiR_fwdQ' \
  model.kwargs.sf_hidden_sizes=null \
  model.kwargs.sf_grad_to_phi=False \
  model.kwargs.reward_grad_to_phi=True \
  training.seed=[0] \
