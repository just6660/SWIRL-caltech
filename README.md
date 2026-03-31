# SWIRL — State-space Inference with Reward Learning

> This codebase is a Caltech mouse behavior adaptation built on top of the original SWIRL framework by the BRAINML-GT lab.
> **Original repository:** [https://github.com/BRAINML-GT/SWIRL](https://github.com/BRAINML-GT/SWIRL)

SWIRL learns a hierarchical hidden Markov model (HMM) with a neural-network reward function from animal behavior sequences. It was developed for the [CalMS21 Task 1](https://data.caltech.edu/records/1991) dataset, which labels mouse social behavior into four classes: **Attack**, **Investigation**, **Mount**, and **Other**.

---

## How it works

SWIRL fits a generative model with three components:

| Component | Description |
|-----------|-------------|
| π₀(z₁) | Initial distribution over K discrete hidden states |
| P(z_t \| z_{t-1}, a_{t-1}) | Hidden state transitions conditioned on the previous action |
| π(a_t \| z_t, s_t) | Emission probabilities = softmax(R_θ(s_t) + log P) |

where `R_θ` is a small MLP that produces a K × C reward matrix (K hidden states, C = 4 behavior classes). The model is fit with Expectation-Maximization:

- **E-step** — forward-backward algorithm over each trajectory → expected state occupancies γ and transition counts ξ
- **M-step** — update π₀ analytically; update P and R_θ by gradient descent (Adam or L-BFGS via JAXOpt)

Training runs in two phases:
1. **Phase 1 (50 iters):** reward network only, transitions fixed
2. **Phase 2 (50 iters):** full joint EM

Two network variants are trained in parallel:
- **S-1 (net1):** first-order, C-dim one-hot input
- **S-2 (net2):** second-order, C²-dim expanded input

---

## Repository layout

```
SWIRL/
├── .gitattributes                  # Git LFS rules for large JSON files
├── requirements.txt                # Python dependencies
└── caltech/
    ├── data/
    │   ├── task1_classic_classification/   # Raw CalMS21 JSON (Git LFS)
    │   ├── seqs.npy                        # Processed full sequences
    │   ├── trans_probs.npy                 # Deterministic transition matrix (4×4×4)
    │   ├── compressed_seqs.npy             # Self-transition-compressed sequences
    │   ├── compressed_trans_probs.npy
    │   └── *_arhmm_caltech*.npz            # Initialization files per K/seed
    ├── results/                            # All training outputs and plots
    └── swirl/
        ├── swirl_func.py                   # Core JAX algorithms (HMM, EM steps)
        ├── caltech_models.py               # Flax MLP reward network
        ├── caltech_analysis.py             # Reward extraction utilities
        ├── process_data.py                 # Preprocess raw JSON → .npy + init .npz
        ├── process_compressed_data.py      # Same but removes self-transitions first
        ├── run_caltech.py                  # EM training on full sequences
        ├── run_caltech_compressed.py       # EM training on compressed sequences
        ├── analyze_caltech.py              # Post-training analysis & figures
        ├── analyze_caltech_compressed.py   # Same for compressed pipeline
        └── plot_accuracy_multiK.py         # Predictive accuracy bar charts
```

---

## Installation

```bash
# Clone the repo (requires Git LFS for the raw data files)
git lfs install
git clone <repo-url>
cd SWIRL

# Install dependencies
pip install -r requirements.txt
```

> **Note:** the requirements file pins JAX 0.4.33 with CUDA 12. If running CPU-only, replace the JAX line with `jax[cpu]`.

---

## Git LFS

Large raw data files are stored with [Git LFS](https://git-lfs.github.com/). The `.gitattributes` file tracks all `.json` files under `caltech/data/task1_classic_classification/`:

```
caltech/data/task1_classic_classification/*.json filter=lfs diff=lfs merge=lfs -text
```

**Setup LFS before your first clone or pull:**

```bash
git lfs install          # one-time setup per machine
git lfs pull             # download the actual file contents after clone
```

When adding new large files (numpy arrays, JSON datasets, etc.) to be tracked:

```bash
git lfs track "caltech/data/*.npy"    # add a new pattern
git add .gitattributes
git add caltech/data/my_file.npy
git commit -m "add processed data"
```

---

## Step-by-step usage

### 1. Download raw data

Place the CalMS21 Task 1 JSON files into:

```
caltech/data/task1_classic_classification/
  calms21_task1_train.json
  calms21_task1_test.json
  taskprog_features_task1_train.json
  taskprog_features_task1_test.json
```

### 2. Preprocess data

```bash
cd caltech/swirl

# Full sequences (2000 frames per video)
python process_data.py <K> <seed>
# e.g.: python process_data.py 2 30

# Compressed sequences (self-transitions removed)
python process_compressed_data.py <K> <seed>
```

**Outputs** (written to `caltech/data/`):
- `seqs.npy` — integer behavior sequences, shape `(n_videos, 2000)`
- `trans_probs.npy` — deterministic 4×4×4 transition tensor
- `<K>_<seed>_arhmm_caltech.npz` — initialization arrays (π₀, P, R)
- Compressed variants of the above with `compressed_` prefix

### 3. Train the model

```bash
# Full sequences
python run_caltech.py <K> <seed>
# e.g.: python run_caltech.py 2 30

# Compressed sequences
python run_caltech_compressed.py <K> <seed>
```

**Outputs** (written to `caltech/results/`):
- `<K>_<seed>_NM_caltech_net1.npz` — trained S-1 model (π₀, P, reward params, LL curves, Viterbi states)
- `<K>_<seed>_NM_caltech_net2.npz` — trained S-2 model
- Compressed variants with `compressed_` prefix

### 4. Analyze and plot results

```bash
# Generate plots for a single K/seed run
python analyze_caltech.py <K> <seed>
python analyze_caltech_compressed.py <K> <seed>

# Compare predictive accuracy across multiple K values
python plot_accuracy_multiK.py <seed>
# e.g.: python plot_accuracy_multiK.py 30
```

**Outputs** (written to `caltech/results/`):
- `<K>_<seed>_training_curves.pdf` — log-likelihood vs. EM iteration
- `<K>_<seed>_Rsa_net{1,2}.pdf` — reward heatmaps per hidden state
- `<K>_<seed>_segments.svg` — hidden state segmentation raster
- `multiK_<seed>_comparison.pdf` — BIC and LL across K values
- `compressed_multiK_<seed>_accuracy.pdf` — predictive accuracy bar chart

---

## File reference

### `swirl_func.py` — core algorithms (JAX)

| Function | Description |
|----------|-------------|
| `forward()` | HMM forward pass; returns log-normalizers and filtered beliefs |
| `backward()` | HMM backward pass |
| `expected_states()` | Computes γ (state occupancy) and ξ (transition expectations) from forward/backward |
| `_viterbi_JAX()` | Viterbi decoding; returns most likely hidden state sequence |
| `jax_soft_find_policy()` | Soft policy via log-sum-exp |
| `vinet()` | Emission log-probs from first-order network (S-1) |
| `vinet_expand()` | Emission log-probs from second-order network (S-2) |
| `comp_ll_jax()` | Log-likelihood of observed actions given states and reward params |
| `comp_transP()` / `comp_log_transP()` | Action-conditioned transition probabilities |
| `pi0_m_step()` | M-step: update initial state distribution analytically |
| `trans_m_step_jax_jaxopt()` | M-step: update transition params via L-BFGS |
| `trans_m_step_jax_optax()` | M-step: update transition params via Adam |
| `emit_m_step_jaxnet_optax2()` | M-step: train reward network for S-1 |
| `emit_m_step_jaxnet_optax2_expand()` | M-step: train reward network for S-2 |
| `jaxnet_e_step_batch()` | E-step batched across trajectories (S-1) |
| `jaxnet_e_step_batch2()` | E-step batched across trajectories (S-2) |

### `caltech_models.py` — Flax neural networks

| Symbol | Description |
|--------|-------------|
| `MLP` | `Dense(hidden) → LeakyReLU → Dense(K × C)`; `expand=True` doubles input dim for S-2 |
| `create_train_state()` | Initializes Flax `TrainState` with Adam optimizer |

### `caltech_analysis.py` — reward extraction

| Function | Description |
|----------|-------------|
| `get_reward_nm()` | Extracts K × C reward matrix from a trained S-1 (net1) network |
| `get_reward_m()` | Same for S-2 (net2) with expanded input |
| `zscore()` | Z-score normalization helper |

---

## Running multiple K values

To compare models with different numbers of hidden states, run steps 2–4 for each K:

```bash
for K in 1 2 3 4; do
    python process_data.py $K 30
    python run_caltech.py $K 30
    python analyze_caltech.py $K 30
done
python plot_accuracy_multiK.py 30
```

`analyze_caltech.py` automatically generates a multi-K comparison PDF when results for multiple K values exist in `caltech/results/`.

---

## Hardware notes

- Training uses JAX with GPU (CUDA 12) by default. Set `JAX_PLATFORM_NAME=cpu` to force CPU.
- Full sequences (T=2000) are memory-intensive for large K. Compressed sequences run significantly faster.
- Results are deterministic given the same `seed` argument.
