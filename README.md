## Description

Tracks first order gradient correlations across layers of any neural network optimisation occurs under Muon and Adam. The measure was first introduced in [Exploring Gradient Oscillation in Deep Neural Network Training](https://ieeexplore-ieee-org.uplib.idm.oclc.org/document/10313361). This uncovers stable or oscillatory gradient phases which are then mappable to various loss and accuracy performance benchmarks adapted from [How Muon’s Spectral Design Benefits Generalization: A Study on Imbalanced Data](https://proceedings.iclr.cc/paper_files/paper/2026/file/e6231c5f46598cfd09ff1970524e0436-Paper-Conference.pdf). The scientific paper designs and/or uses common benchmarks to study the inductive learning biases ("rich" or "simple") of Muon and Adam. 

### What is the gradient correlation? 

The gradient correlation (GCORR) simply measures the dot product *then* cosine similarity per layer of (adjacent) gradient vector pairs generated as optimisation occurs. For $n$ train steps this obtains $n-1$ **scalars** per layer. Taking the history for all layers together, a grid is formed. 

During live computation for storage only the past gradient vectors are required in memory. For time taken the GCORR hook takes only 0.1 - 0.2% of train budget (included validation as design choice). 

## Reported Metrics

All reported metrics are over n runs, obtaining a mean, and quantile distributions shaded around it. Options exist to customise metric aggregation incl per epoch and/or per column averaging. An example result of opinionated metrics is shown below:



## Extendability 

How hard is it to adopt and browse the repo for related experiments?


## 


 Measure used stability or oscillations. This is done per layer before aggregation to global metrics. 
- Uncovers 