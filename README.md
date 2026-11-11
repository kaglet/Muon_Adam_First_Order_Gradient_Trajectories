## Description

Tracks first order gradient correlations across layers of any neural network optimisation occurs under Muon and Adam. The measure was first introduced in [Exploring Gradient Oscillation in Deep Neural Network Training](https://ieeexplore-ieee-org.uplib.idm.oclc.org/document/10313361). This uncovers stable or oscillatory gradient phases which are then mappable to various loss and accuracy performance benchmarks adapted from [How Muon’s Spectral Design Benefits Generalization: A Study on Imbalanced Data](https://proceedings.iclr.cc/paper_files/paper/2026/file/e6231c5f46598cfd09ff1970524e0436-Paper-Conference.pdf). The scientific paper designs and/or uses common benchmarks to study the inductive learning biases ("rich" or "simple") of Muon and Adam. 

### What is the gradient correlation? 

The gradient correlation (GCORR) simply measures the (linear algebra) dot product *then* cosine similarity per layer of (adjacent) gradient vector pairs generated as optimisation occurs. For $n$ train steps this obtains $n-1$ **scalars** per layer. Taking the history for all layers together, a grid is formed. 

During live computation for storage only the past gradient vectors are required in memory. For time taken the GCORR hook takes only 0.1 - 0.2% of train budget (included validation as design choice). 

## Experiments Possible

These are the finished pipelines. 

### Class Imbalance Small and Large on Residual Networks

This is a hybrid mix that extends both studies of [Exploring Gradient Oscillation in Deep Neural Network Training](https://ieeexplore-ieee-org.uplib.idm.oclc.org/document/10313361) and [How Muon’s Spectral Design Benefits Generalization: A Study on Imbalanced Data](https://proceedings.iclr.cc/paper_files/paper/2026/file/e6231c5f46598cfd09ff1970524e0436-Paper-Conference.pdf).

Ratios := $R={20,10,5,1}$ define downsampling ratios where half classes are maximised and half are downsampled by the factor of $R$. 

- Class Imbalance "Small" = CIFAR10 + ResNet18 (default random weight init) + 50 epochs 
- Class Imbalance "Large" = CIFAR100 + ResNet50 (default random weight init) + 50 epochs

## Reported Metrics

All reported metrics are over n runs, obtaining a mean, and quantile distributions shaded around it. Options exist to customise metric aggregation incl per epoch and/or per column averaging. Example results of opinionated metrics used to support our own study are shown below:

### Lone Plots

![Output Example](outputs/cls_imb_s/report/mean/R_20/lone_muon_and_adam_cosine_similarity.png) 

![Output Example](outputs/cls_imb_s/report/mean/R_20/lone_muon_and_adam_dot_product.png) 

![Output Example](outputs/cls_imb_s/report/mean/R_20/lone_muon_and_adam_train_and_validation_accuracy.png) 

![Output Example](outputs/cls_imb_s/report/mean/R_20/lone_muon_and_adam_train_and_validation_loss.png) 

![Output Example](outputs/cls_imb_s/report/mean/R_20/lone_subgroup_muon_and_adam_train_and_validation_accuracy.png)


## Grid Plots

[![](outputs/cls_imb_s/report/mean/R_20/grid_per_layer_muon_and_adam_cosine_similarity.png)](outputs/cls_imb_s/report/mean/R_20/grid_per_layer_muon_and_adam_cosine_similarity.png) 

[![](outputs/cls_imb_s/report/mean/R_20/grid_per_layer_muon_and_adam_dot_product.png)](outputs/cls_imb_s/report/mean/R_20/grid_per_layer_muon_and_adam_dot_product.png) 

## Usage 

To replicate the current available experiments without issue the only hardware requirement is use of windows and a single GPU. Alternate instructions are present for runs on a linux cluster such as CHPC (also used):

(1) Install [pixi](https://pixi.prefix.dev/latest/installation/) for env management = 2-5min.
(2) Install [git](https://git-scm.com/install/windows) = 2-5min.
(3) With GitHub, fork and clone this repo. Or download as zip and unpack. 
(4) In any terminal `cd` to within project root directory. 
(5) Run `pixi install`. (On chpc run `CONDA_OVERRIDE_CUDA=12.0 pixi install`)
(6) Run `pixi run grid_run_local`. (On chpc run `CONDA_OVERRIDE_CUDA=12.0 pixi run grid_run_pbs`). This trains over default ratios $[20,1]$ for the `"small"` experiment by default. 
    - Edit the `override_strings.py` in the `/small` subfolder to add any number of ratios in the downsample ratio list. 
    - To train `"large"` alone or with `"small"` a pre-existing variable and `if else` block makes this trivial in the `grid_run_local` module.
(7) With project root as working directory: Inspect results of each run stored in `"./outputs/cls_imb_s"` and `"./outputs/cls_imb_l"` for small and large respectively in a nested structure. 
(8) Run `pixi run summarise_metrics` to generate the default aggregate metrics over $n$ runs. Metrics are outputted in a flat list one folder above the $n$ run subfolders. 
    - This step is highly efficient and uses pandas grouping. Expect well under a minute for "small" and "large" separately. 
(9) Run `viz_report_metrics` to generate png plot images in `outputs\cls_imb_{s | l}\report` for all ratios. This will yield outputs as shown prior. 
    - This step is slow when it generates the grids. Allow 3-5min per grid as stated in terminal output. 
(10) (Optional) Run `post_proc_convert` to use a helper function that cleans up all aggregate metrics and leaves the core run metrics untouched.  

## Core Open Source Frameworks 

This list may not yet be exhaustive. Other credits for smaller snippets where necessary are embedded in code with `@<username>`. This may be pytorch community member or github usernames. This will definitely be updated as is deserving. 

- [Hydra-Zen](https://github.com/mit-ll-responsible-ai/hydra-zen): Uses Hydra -> OmegaConf in a pythonic form which made the ML training (or any pythonic setup) incredibly configurable, readable, and alterable beyond defaults. 
- [Pytorch Lightning (PL)](https://github.com/lightning-ai/pytorch-lightning): Forced use of decoupled components and hooks. 
- [Muon](https://github.com/KellerJordan/Muon) by its author [Keller Jordan](https://github.com/KellerJordan): For the dual optimiser class of Muon and Adam as a single PyTorch wrapper `MuonAuxAdam` optimiser.
- [Dual Optimiser for PL](https://github.com/shivvor2/research-monorepo/blob/master/src/research_lib/training/modules/dual_optimizer.py) by [@shivvor2](https://github.com/shivvor2): Initially used as basis for workarounds needed to make dual optimisers work in PL if certain custom optimisations are desired. The code is left as is for future users who would wish to add other optimisers beyond Muon or Adam `:)`.

## If you'd like to use

If you have are interested in the repo, feel free to contact and ask. This was designed with libraries/tools that encourage replicability and expanded usage patterns for the paradigm of research studies adapted from. The modular and configurable structure means isolated usage and adaptation of components is possible, and can be polished even further. 

However (!) recommendations for future adaptation include: 
- PyTorch Fabric for more complex experiments and scientific research, carrying over lessons learned in modularity from PLightning. 
- Hydra-Zen usage is definitely recommended for configurability that scales with complexity. 

Please keep the below in mind...

Admittedly the most confusing pipeline is the `viz_report_metrics` module. It uses pure matplotlib and some tricks to add more efficiency, control and reuse of fig/ax components to obtain a desired yet opinionated reporting result. Small things such as renaming confusing variable names may definitely assist clarity. 

## Future Plans

*To be updated soon...*