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
## Appendix (Theory part of the proposed method)
\subsection{Proposition (false-negative reduction) and proof sketch}
\textbf{Proposition.} \emph{(False-negative penalty reduction.)} Assume $\widehat{A}_{ij}$ is positively correlated with the indicator $\mathbf{1}\{Y_i=1, Y_j=1\}$ of true co-occurrence for class pair $(i,j)$. If the metric loss tends to push apart unlabeled co-occurring clips (treating them as negatives), then adding LIR with weight $\widehat{A}_{ij}$ reduces, in expectation, the embedding dissimilarity between clips that truly co-occur, thereby lowering the expected contribution of false-negative pairs to the metric loss.

\textbf{Proof sketch.} LIR applies a negative gradient on $D(z_n,z_m)$ proportional to $\widehat{A}_{ij}$ for pairs $(n,m)$ in $\mathcal{S}_{ij}$. Under the alignment assumption, pairs that are true co-occurrences receive larger $\widehat{A}_{ij}$ and thus larger pull in embedding space; this reduces their expected dissimilarity $\mathbb{E}[D]$, which in turn reduces the metric-loss terms attributable to false negatives (the portion of metric loss sensitive to large dissimilarities on true co-occurrences). The argument relies on the teacher providing weights correlated with true co-occurrence; we validate this empirically (Section~X).

\subsection{Empirical consistency checks and metrics}
To replace strong formal convergence claims we recommend and report the following empirical tests:

\paragraph{ILI stability (bootstrap / seeds).} Repeat the two-stage pipeline for $R$ runs (different seeds or bootstrap resamples). Report (i) edge frequency $\text{freq}_{ij}=\frac{1}{R}\sum_r\mathbf{1}\{\widehat{A}_{ij}^{(r)}>0\}$ for top edges, and (ii) average Frobenius distance between adjacency matrices to quantify stability.

\paragraph{False-negative embedding-distance test.} For each top edge $(i,j)$ define
\[
\mathcal{P}_{ij}=\{(n,m)\,:\,s_n[i]>\tau,\,s_m[j]>\tau\}.
\]
Compute mean dissimilarities without and with LIR:
\[
\overline{D}^{\text{noLIR}}_{ij},\quad \overline{D}^{\text{LIR}}_{ij},
\]
and report $\Delta\overline{D}_{ij}=\overline{D}^{\text{noLIR}}_{ij}-\overline{D}^{\text{LIR}}_{ij}$ (positive $\Rightarrow$ LIR pulled co-activated pairs closer). Aggregate over top edges (mean, 95\% CI).

\paragraph{MAP-over-epochs.} Plot retrieval MAP vs epoch for baseline (no LIR) and LIR runs (LIR starting at $E_{\text{teach}}$). Use MAP curves to show LIR's effect on convergence and generalization.

\subsection{Implementation notes}
\begin{itemize}
  \item Use independent sigmoid heads for per-class soft-labels and temperature / calibration (e.g., Platt scaling or temperature tuning) to stabilize $\widehat{A}$.
  \item Restrict LIR to top-$K$ edges per node and normalize $\widehat{A}$ for numerical stability.
  \item Approximate expectations in Eq.~\eqref{eq:ccr} with intra-batch sampled pairs; for efficiency sample a fixed number of pairs per edge.
  \item Set $\lambda$ and $K$ via validation; report ablations for $E_{\text{teach}}\in\{300,400,\dots,900\}$.
\end{itemize}

\subsection{Limitations}
The ILI adjacency $\widehat{A}$ is inferred from teacher \emph{soft-labels} and therefore reflects discovered statistical dependencies rather than proven causal relations; ground-truth causation would require interventional data. LIR depends on teacher calibration and stability; to mitigate spurious edges we (i) calibrate per-class outputs, (ii) restrict LIR to high-weight/stable edges, (iii) report bootstrap stability, and (iv) validate edges with small perturbation tests and case studies. These diagnostics are reported alongside retrieval metrics (MAP) and embedding-distance analyses.
