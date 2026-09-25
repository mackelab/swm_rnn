# Attractor dynamics in data-constrained recurrent neural networks of sequence working memory

---

Repository accompanying [biorxiv (still empty)]()

---
### Overview 

Code for fitting **stochastic low-rank RNNs** to **single-unit spike recordings** from the sequence working memory (SWM) task of [Chen et al., *Neuron* 2024](https://doi.org/10.1016/j.neuron.2024.07.024), and for generating the associated analysis figures.

Note that the variational SMC training pipeline is largely that of [**smc_rnns**](https://github.com/mackelab/smc_rnns) ([Pals et al., *NeurIPS* 2024](https://arxiv.org/abs/2406.16749)). This repository adapts that framework to multi-session datasets. If you want to start using this method for training RNNs, a good place to start is likely the [tutorial notebooks](https://github.com/mackelab/smc_rnns/tree/main/tutorial) in the [**smc_rnns**](https://github.com/mackelab/smc_rnns)  repo.

---

### Environment and training

Create the project environment:

```bash
conda env create -f swm_rnn_env_cuda.yml
```

or if on a mac / no gpu available: 

```bash
conda env create -f swm_rnn_env.yml
```

and activate it:

```bash
conda activate swm_rnn_env
```

Notebooks add the repo root to `sys.path` automatically. For training scripts, run from the repo root:

```bash
python train_scripts/train_swm.py
```

---

### Data

Download the dataset of [Chen et al., *Neuron* 2024](https://doi.org/10.1016/j.neuron.2024.07.024), from [Zenodo](https://doi.org/10.5281/zenodo.13119704) (this requires a request to the authors), and arrange session folders under:

```
data/
├── data_groot/
│   └── Data_forward/
│       └── groot2024-03-11/   # one folder per session
│       └── ...
└── data_ocean/
    └── Data_forward/
        └── ocean2021-08-20/
        └── ...
```

Each session folder should contain `Neuron.mat` and `TrialInfo.mat`. 

Session names must match those listed in [`train_scripts/train_swm.py`](train_scripts/train_swm_bs.py).

---

### Reproducing figures

Most analysis notebooks default to `run = False`, which loads cached `data/df_*.pkl` files and skips expensive recomputation. Set `run = True` in the controls cell to recompute from `final_models/` and raw data. For the latter the notebooks should be run in order.

Shared plotting and analysis helpers live in [`generate_figures/fig_utils/`](generate_figures/fig_utils/).


