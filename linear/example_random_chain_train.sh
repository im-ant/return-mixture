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

python -u train_linear_prediction.py \
    hydra.run.dir=$sweep_parent_dir \
    hydra.sweep.dir=$sweep_parent_dir \
    training.num_episodes=400 \
    training.save_checkpoint=null \
    training.seed=[2,4,6,8,10,12,14,16,18,20] \
    env=random_walk_chain \
    agent=sf_return \
    agent.kwargs.lr=[0.01,0.1,0.2,0.3,0.5,0.8,1.0] \
    agent.kwargs.gamma=1.0 \
    agent.kwargs.lamb=[0.0,0.3,0.5,0.7,0.9,0.99,1.0] \
    agent.kwargs.eta_trace=0.0 \
    agent.kwargs.use_true_sf_params=[False] \
    agent.kwargs.use_true_reward_params=[False] \
