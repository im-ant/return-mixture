# η-Return Mixture

This is a PyTorch implementation of the η-return mixture idea from our AAAI 2022 paper:

> GX-Chen, Anthony, Veronica Chelu, Blake A. Richards, and Joelle Pineau. "A Generalized Bootstrap Target for Value-Learning, Efficiently Combining Value and Feature Predictions." _arXiv preprint arXiv:2201.01836 (2022)_.

[[arXiv](https://arxiv.org/abs/2201.01836)]

## Citation

```
@article{gxchen2022generalized,
  title={A Generalized Bootstrap Target for Value-Learning, Efficiently Combining Value and Feature Predictions},
  author={GX-Chen, Anthony and Chelu, Veronica and Richards, Blake A and Pineau, Joelle},
  journal={arXiv preprint arXiv:2201.01836},
  year={2022}
}
```

## Note

#### Main results

- Deterministic chain result can be generated in notebook
  ```
  notebook/Deterministic-Chain-Example.ipynb
  ```
  - This is a self-contained notebook and lets you play with the return mixture idea for value prediction

- Random walk chain experiment can be generated via example
  ```
  linear/example_random_chain_train.sh
  ```
- An example for how to run Mini-Atari with η-return mixture augmented DQN is provided in example
  ```
  nonlinear/example_lvf_q_learning.sh
  ```
- The MinAtar training result can be generated via example (can be run locally or submitted via SLURM)
  ```
  nonlinear/example_grid_sweep_job.py
  ```
- The learning rate parameter study (figure 4) can be generated by changing the learning rate in the above examples
