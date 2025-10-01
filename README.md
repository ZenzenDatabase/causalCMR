# Causal Audio–Visual Embedding Learning

## Overview  
This repository contains the implementation of our causality-driven framework for learning robust audio–visual embeddings. We combine (1) Audio–Visual Semantic Alignment Loss (AV-SAL), (2) causal structure discovery via GRaSP, and (3) Causal Consistency Regularization (CCR) to disentangle true semantic dependencies from incidental co-occurrences.

## Features  
- Teacher–student training schedule with configurable transition epoch  
- AV-SAL for cross-modal soft-label alignment  
- GRaSP algorithm for directed causal graph inference  
- CCR loss for embedding regularization based on discovered causal links  
- Proxy-based triplet loss integration

## Requirements  
- Python 3.10+  
- PyTorch 1.12+  
- NumPy, SciPy, scikit-learn  
- GRaSP dependency (see `requirements.txt`)  
- CUDA-enabled GPU recommended

## Installation  
```bash
git clone https://github.com/me/causal-audio-visual-embedding.git
cd causal-audio-visual-embedding
pip install -r requirements.txt

More information will be upload soon.
```
## Appendix: Theoretical Details of the Proposed Method

### Proposition: False-Negative Penalty Reduction

**Proposition**: *False-Negative Penalty Reduction.*  
Assume $\widehat{A}_{ij}$ is positively correlated with the true co-occurrence indicator $\mathbf{1}\{Y_i=1, Y_j=1\}$ for class pair $(i,j)$. If the metric loss pushes apart unlabeled co-occurring clips (treating them as negatives), adding Label-Imputation Regularization (LIR) with weight $\widehat{A}_{ij}$ reduces, in expectation, the embedding dissimilarity between truly co-occurring clips, lowering the contribution of false-negative pairs to the metric loss.

**Proof Sketch**:  
LIR applies a negative gradient on the embedding distance $D(z_n, z_m)$ proportional to $\widehat{A}_{ij}$ for pairs $(n,m)$ in the co-occurrence set $\mathcal{S}_{ij}$. Under the correlation assumption, true co-occurring pairs receive larger $\widehat{A}_{ij}$, pulling their embeddings closer in expectation. This reduces the expected dissimilarity $\mathbb{E}[D]$, decreasing the metric loss from false negatives (terms sensitive to large dissimilarities for true co-occurrences). The teacher’s weights are empirically validated to correlate with true co-occurrences (see Section X).

### Empirical Consistency Checks and Metrics

To ensure robustness without strong formal convergence guarantees, we recommend the following empirical evaluations:

#### ILI Stability (Bootstrap/Seeds)  
Run the two-stage pipeline $R$ times with different seeds or bootstrap resamples. Report:  
- **Edge Frequency**: $\text{freq}_{ij} = \frac{1}{R} \sum_r \mathbf{1}\{\widehat{A}_{ij}^{(r)} > 0\}$ for top edges.  
- **Adjacency Stability**: Average Frobenius distance between adjacency matrices to quantify consistency.

#### False-Negative Embedding-Distance Test  
For each top edge $(i,j)$, define the co-activated pair set:  
\[
\mathcal{P}_{ij} = \{(n,m) : s_n[i] > \tau, s_m[j] > \tau\}.
\]
Compute mean embedding dissimilarities with and without LIR:  
\[
\overline{D}^{\text{noLIR}}_{ij}, \quad \overline{D}^{\text{LIR}}_{ij},
\]
and report the difference:  
\[
\Delta\overline{D}_{ij} = \overline{D}^{\text{noLIR}}_{ij} - \overline{D}^{\text{LIR}}_{ij}.
\]
Positive $\Delta\overline{D}_{ij}$ indicates LIR reduces distances for co-activated pairs. Aggregate results over top edges (mean, 95% CI).

#### MAP-over-Epochs  
Plot Mean Average Precision (MAP) vs. epoch for baseline (no LIR) and LIR runs (LIR starts at epoch $E_{\text{teach}}$). Use MAP curves to evaluate LIR’s impact on convergence and generalization.

### Implementation Notes

- **Soft Labels**: Use independent sigmoid heads for per-class soft labels, stabilized with temperature tuning or Platt scaling for reliable $\widehat{A}$.
- **Edge Selection**: Limit LIR to top-$K$ edges per node and normalize $\widehat{A}$ for numerical stability.
- **Sampling**: Approximate expectations in the co-occurrence regularization term (Eq. X) using intra-batch sampled pairs; sample a fixed number of pairs per edge for efficiency.
- **Hyperparameters**: Tune $\lambda$ and $K$ on a validation set. Report ablations for $E_{\text{teach}} \in \{300, 400, \dots, 900\}$.
\subsection{Limitations}
The ILI adjacency $\widehat{A}$ is inferred from teacher \emph{soft-labels} and therefore reflects discovered statistical dependencies rather than proven causal relations; ground-truth causation would require interventional data. LIR depends on teacher calibration and stability; to mitigate spurious edges we (i) calibrate per-class outputs, (ii) restrict LIR to high-weight/stable edges, (iii) report bootstrap stability, and (iv) validate edges with small perturbation tests and case studies. These diagnostics are reported alongside retrieval metrics (MAP) and embedding-distance analyses.
