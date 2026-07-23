"""
ARCHITECTURE: MULTI-OUTPUT TRANSFORMER (TRF)
=============================================

The TRF model solves a joint pricing problem for k=4 targets simultaneously:

    freq_property  -- property damage claim count   (Poisson)
    freq_liability -- bodily-injury claim count     (Poisson)
    sev_property   -- average property severity     (Gamma)
    sev_liability  -- average liability severity    (Gamma)

Key innovations over a plain MLP:

1. TARGET-SPECIFIC EMBEDDINGS
   Every covariate x_j is embedded independently for each target t:

       e_{t,j} = Emb^{(t,j)}(x_j)

   • Categorical x_j  →  entity embedding lookup  (per-target table)
   • Continuous x_j   →  piecewise-linear encoding → Linear(embed_dim)

   This lets "driver age" influence property frequency differently from
   how it influences liability severity — each target "sees" the same
   covariate through its own learned lens.

   All feature embeddings for target t are concatenated:

       h_t  =  [e_{t,1}, ..., e_{t,m}]  ∈ R^d       d = m · embed_dim

   A shallow FFN then produces an initial scalar prediction ŷ_t^(FFN):

       ŷ_t^(FFN)  =  FFN_t(h_t)                       scalar ∈ R

2. TRANSFORMER AUGMENTATION (cross-target attention)
   The h_t vectors are treated as "tokens" — one token per target.
   A CLS token h_CLS is appended:

       H' = [h_1, h_2, h_3, h_4, h_CLS]  ∈ R^{(k+1) × d}

   Multi-head self-attention over H' lets every target attend to all
   others, learning which covariate signals are shared across guarantees:

       Z' = Transformer(H')  ∈ R^{(k+1) × d}

   The CLS token z_CLS = Z'[:, -1, :] aggregates cross-target information.

3. FINAL PREDICTION
   A small FFN maps z_CLS to a k-dimensional augmentation vector a:

       a  =  FFN_CLS(z_CLS)  ∈ R^k

   This is added to the stacked initial predictions and passed through σ:

       p_aug = σ(a + [ŷ_1^(FFN), ..., ŷ_k^(FFN)])   ∈ (0,1)^k

   • Poisson targets:  ŷ_t  =  p_aug,t × exposure   (expected count)
   • Gamma targets:    ŷ_t  =  p_aug,t              (scaled severity ∈ (0,1))

DEPENDENCY STRUCTURE (HYPOTHESIS)
==================================
We hypothesize a GAUSSIAN COPULA linking the four latent log-rates:

    Z = [log λ_freq_prop, log λ_freq_liab, log μ_sev_prop, log μ_sev_liab]

with correlation matrix:

    Σ = [[1.00, 0.40, 0.15, 0.10],
         [0.40, 1.00, 0.10, 0.20],
         [0.15, 0.10, 1.00, 0.30],
         [0.10, 0.20, 0.30, 1.00]]

Economic rationale
------------------
• ρ(freq_prop, freq_liab) = 0.40 — "shared frailty": risky drivers
  (impulsive behaviour, distraction) generate BOTH material damage and
  bodily-injury claims. This latent tendency is not captured by observable
  rating factors. At-fault accidents create both guarantees simultaneously.

• ρ(sev_prop, sev_liab) = 0.30 — accident severity: a violent crash
  causes BOTH large repair bills and serious bodily injury simultaneously.

• Cross (freq ↔ sev) = 0.10–0.20 — mild positive dependence since all
  four targets are driven by the same underlying accident process.

The copula is implemented by adding correlated Gaussian noise to each
deterministic log-rate: Z_eff = Z_det + ε,  ε ~ N(0, Σ_cov).

COVARIATES
----------
Variable    Type         Description
---------   ----------   ---------------------------------------------------
age         continuous   Driver age, years (18–80)
sex         categorical  M / F
race        categorical  Anonymised groups A1–A5 (no ranking implied)
exposure    continuous   Fraction of year insured (0.1–1.0)
carage      continuous   Vehicle age in years (0–20)
category    categorical  Vehicle type: Sedan, SUV, Pickup, Motorcycle, Van
"""

# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CONFIGURATION
# =============================================================================
N_POLICIES  = 50_000
N_BINS      = 10        # piecewise-linear encoding bins (per continuous var)
EMBED_DIM   = 8         # embedding dimension per feature per target
N_HEADS     = 2         # transformer attention heads
DROPOUT     = 0.10
BATCH_SIZE  = 512
MAX_EPOCHS  = 80
PATIENCE    = 20        # early stopping

TARGET_NAMES = ["freq_property",  "freq_liability",
                "sev_property",   "sev_liability"]
TARGET_TYPES = ["poisson", "poisson", "gamma", "gamma"]

CAT_VARS  = ["sex", "race", "category"]
CONT_VARS = ["age", "carage"]            # exposure handled separately


# =============================================================================
# SECTION 1 – DATA SIMULATION
# =============================================================================

def simulate_auto_portfolio(n: int = N_POLICIES, seed: int = SEED) -> pd.DataFrame:
    """
    Simulate an auto insurance portfolio with property and liability guarantees.

    Dependency structure
    --------------------
    Correlated Gaussian noise ε ~ N(0, Σ_cov) is added to each deterministic
    log-rate so that the *effective* log-rates share the Gaussian copula
    defined by Σ (correlation matrix above).

    Rating factors (deterministic part)
    ------------------------------------
    Property frequency:
        Elevated for young/senior drivers, older cars, motorcycles, pickups.
    Liability frequency:
        Elevated for young males, motorcycles; driven by risky behaviour.
    Property severity:
        Higher for newer/expensive cars (SUV, Motorcycle), lower for old cars.
    Liability severity:
        Higher for young drivers (more serious injuries), motorcycles.
    """
    rng = np.random.default_rng(seed)

    # ── Covariates ────────────────────────────────────────────────────────────
    age      = rng.integers(18, 81, n).astype(float)
    sex      = rng.choice(["M", "F"], n, p=[0.55, 0.45])
    # Anonymised demographic groups — no ranking implied
    race     = rng.choice(["A1", "A2", "A3", "A4", "A5"], n,
                           p=[0.30, 0.25, 0.20, 0.15, 0.10])
    exposure = np.round(rng.uniform(0.1, 1.0, n), 2)
    carage   = rng.integers(0, 21, n).astype(float)
    category = rng.choice(
        ["Sedan", "SUV", "Pickup", "Motorcycle", "Van"], n,
        p=[0.40, 0.25, 0.15, 0.10, 0.10],
    )

    # ── Deterministic log-rates ───────────────────────────────────────────────
    log_lam_prop = (
        -2.80
        + np.where(age < 25, 0.50, 0.0)
        + np.where(age > 65, 0.15, 0.0)
        + 0.010 * carage
        + np.where(sex == "M", 0.10, 0.0)
        + np.where(category == "Motorcycle", 0.60, 0.0)
        + np.where(category == "Pickup",     0.25, 0.0)
        + np.where(category == "Van",        0.10, 0.0)
        + np.where(race == "A1", 0.05, 0.0)
        + np.where(race == "A5", -0.10, 0.0)
    )

    log_lam_liab = (
        -3.20
        + np.where(age < 25, 0.70, 0.0)
        + np.where(age > 65, 0.20, 0.0)
        + np.where(sex == "M", 0.20, 0.0)
        + 0.005 * carage
        + np.where(category == "Motorcycle", 0.80, 0.0)
        + np.where(category == "SUV",        0.15, 0.0)
        + np.where(category == "Pickup",     0.20, 0.0)
        + np.where(race == "A1", 0.03, 0.0)
        + np.where(race == "A5", -0.05, 0.0)
    )

    # Property severity: newer / larger vehicles cost more to repair
    log_mu_prop_sev = (
        7.50
        - 0.015 * carage
        + np.where(category == "SUV",        0.30, 0.0)
        + np.where(category == "Motorcycle", 0.20, 0.0)
        + np.where(category == "Pickup",     0.15, 0.0)
    )

    # Liability severity: injury cost driven by driver age & vehicle type
    log_mu_liab_sev = (
        8.00
        + np.where(age < 25,  0.35, 0.0)
        + np.where(age > 65,  0.15, 0.0)
        + np.where(sex == "M", 0.05, 0.0)
        + np.where(category == "Motorcycle", 0.50, 0.0)
        + np.where(category == "SUV",        0.20, 0.0)
    )

    # ── Gaussian copula – correlated latent noise ─────────────────────────────
    # Order: [freq_prop, freq_liab, sev_prop, sev_liab]
    rho = np.array([
        [1.00, 0.40, 0.15, 0.10],
        [0.40, 1.00, 0.10, 0.20],
        [0.15, 0.10, 1.00, 0.30],
        [0.10, 0.20, 0.30, 1.00],
    ])
    sigma = np.array([0.30, 0.30, 0.40, 0.50])          # per-target noise std
    cov   = np.outer(sigma, sigma) * rho                  # covariance matrix
    eps   = rng.multivariate_normal(np.zeros(4), cov, n)  # shape (n, 4)

    # Effective (noisy) log-rates
    lam_prop_eff  = np.exp(log_lam_prop     + eps[:, 0]) * exposure
    lam_liab_eff  = np.exp(log_lam_liab     + eps[:, 1]) * exposure
    mu_prop_eff   = np.exp(log_mu_prop_sev  + eps[:, 2])
    mu_liab_eff   = np.exp(log_mu_liab_sev  + eps[:, 3])

    # ── Claim counts – Poisson ────────────────────────────────────────────────
    num_prop  = rng.poisson(lam_prop_eff).astype(float)
    num_liab  = rng.poisson(lam_liab_eff).astype(float)

    # ── Claim severities – Gamma (per claim) ─────────────────────────────────
    # Gamma(shape=α, scale=μ/α)  → mean=μ, var=μ²/α
    sev_prop  = np.zeros(n)
    sev_liab  = np.zeros(n)

    mask_p = num_prop  > 0
    mask_l = num_liab  > 0

    alpha_prop, alpha_liab = 3.0, 2.0      # dispersion parameters

    if mask_p.any():
        sev_prop[mask_p] = rng.gamma(
            alpha_prop, mu_prop_eff[mask_p] / alpha_prop, mask_p.sum()
        )
    if mask_l.any():
        sev_liab[mask_l] = rng.gamma(
            alpha_liab, mu_liab_eff[mask_l] / alpha_liab, mask_l.sum()
        )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "age":          age,
        "sex":          sex,
        "race":         race,
        "exposure":     exposure,
        "carage":       carage,
        "category":     category,
        "num_property":  num_prop,
        "num_liability": num_liab,
        "sev_property":  sev_prop,      # avg cost per claim (0 if no claim)
        "sev_liability": sev_liab,
    })

    # 70 / 10 / 20 split
    perm = rng.permutation(n)
    groups = np.empty(n, dtype="U8")
    groups[perm[:int(0.70 * n)]] = "train"
    groups[perm[int(0.70 * n):int(0.80 * n)]] = "valid"
    groups[perm[int(0.80 * n):]] = "test"
    df["group"] = groups

    return df


def print_kpis(df: pd.DataFrame) -> None:
    """Print basic actuarial KPIs per guarantee."""
    print("\n  Actuarial KPIs")
    print(f"  {'Guarantee':<14} {'Frequency':>10}  {'Mean severity':>14}  {'n_policies':>10}")
    print("  " + "-" * 54)
    for g in ["property", "liability"]:
        freq = df[f"num_{g}"].sum() / df["exposure"].sum()
        mask = df[f"num_{g}"] > 0
        sev  = df.loc[mask, f"sev_{g}"].mean() if mask.any() else 0.0
        print(f"  {g:<14} {freq:>10.3%}  {sev:>14,.0f}  {len(df):>10,}")

    # Empirical claim-level correlation (dependency validation)
    freq_corr = np.corrcoef(df["num_property"], df["num_liability"])[0, 1]
    print(f"\n  Observed correlation of claim counts: {freq_corr:.3f}  "
          f"(theoretical latent ρ ≈ 0.40)")


# =============================================================================
# SECTION 2 – PYTORCH MODULES
# (Adapted from ron_approach_utils.py — Spedicato & Richman 2025)
# =============================================================================

class PiecewiseLinearEncoding(nn.Module):
    """
    Encode a scalar continuous variable into a d-bin piecewise-linear vector.

    Given fixed bin boundaries [b_0, b_1, ..., b_d] computed from training
    data (e.g., deciles), each output dimension i indicates how far the input
    falls within bin [b_i, b_{i+1}]:

        output_i = clip((x - b_i) / (b_{i+1} - b_i), 0, 1)

    Reference: Gorishniy, Rubachev, Babenko (2022), "On Embeddings for
               Numerical Features in Tabular Deep Learning."
    """

    def __init__(self, boundaries: list):
        super().__init__()
        b = torch.tensor(boundaries, dtype=torch.float32)
        self.register_buffer("_left",  b[:-1].unsqueeze(0))  # (1, n_bins)
        self.register_buffer("_right", b[1:].unsqueeze(0))   # (1, n_bins)
        self.n_bins = len(boundaries) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().reshape(-1, 1)                          # (batch, 1)
        width = self._right - self._left + 1e-8
        return torch.clamp((x - self._left) / width, 0.0, 1.0)
        # output shape: (batch, n_bins)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head self-attention + FFN.

    This is the core of the TRF model.  Adapted from the implementation in
    ron_approach_utils.TransformerLayerMHAnorm_deep_mask.

    forward() returns (output_sequence, cls_token):
        output_sequence : (batch, seq, d)  — full transformed sequence
        cls_token       : (batch, d)       — last token (CLS position)

    The CLS token is extracted AFTER self-attention, so it contains
    information aggregated from all k target tokens.

    Dropout is applied automatically in training mode via model.train() /
    model.eval().
    """

    def __init__(self, model_dim: int, n_heads: int, ffn_dim: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.n_heads   = n_heads
        self.head_dim  = model_dim // n_heads

        # Multi-head attention projections
        self.W_q = nn.Linear(model_dim, model_dim, bias=False)
        self.W_k = nn.Linear(model_dim, model_dim, bias=False)
        self.W_v = nn.Linear(model_dim, model_dim, bias=False)
        self.W_o = nn.Linear(model_dim, model_dim)

        # Feed-forward sublayer
        self.ffn1 = nn.Linear(model_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim,   model_dim)

        # Normalisation + dropout
        self.ln1  = nn.LayerNorm(model_dim, eps=1e-6)
        self.ln2  = nn.LayerNorm(model_dim, eps=1e-6)
        self.dp_a = nn.Dropout(dropout_rate)
        self.dp_f = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq, model_dim)
        """
        batch, seq, _ = x.shape
        d_h = self.head_dim

        # ── Multi-head self-attention ──────────────────────────────────────
        # Split into heads: (batch, n_heads, seq, head_dim)
        Q = self.W_q(x).reshape(batch, seq, self.n_heads, d_h).transpose(1, 2)
        K = self.W_k(x).reshape(batch, seq, self.n_heads, d_h).transpose(1, 2)
        V = self.W_v(x).reshape(batch, seq, self.n_heads, d_h).transpose(1, 2)

        scores  = torch.matmul(Q, K.transpose(-2, -1)) / (d_h ** 0.5)  # (b,h,s,s)
        weights = F.softmax(scores, dim=-1)
        weights = self.dp_a(weights)

        attended = torch.matmul(weights, V)                   # (b, h, s, d_h)
        attended = attended.transpose(1, 2).reshape(batch, seq, self.model_dim)
        attended = self.W_o(attended)

        # Residual + LayerNorm 1
        x = self.ln1(x + attended)

        # ── Feed-forward sublayer ──────────────────────────────────────────
        ffn = self.dp_f(self.ffn2(F.gelu(self.ffn1(x))))

        # Residual + LayerNorm 2
        x = self.ln2(x + ffn)

        # ── Extract CLS token (appended as last position) ─────────────────
        cls_token = x[:, -1, :]                               # (batch, model_dim)
        return x, cls_token


# =============================================================================
# SECTION 3 – LOSS FUNCTIONS
# =============================================================================

def poisson_deviance_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean Poisson deviance:
        D = 2 · E[ŷ − y + y·log(y/ŷ)]
    Handles y=0 via torch.where (convention: 0·log(0) = 0).
    """
    eps     = 1e-7
    y_pred  = y_pred.clamp(min=eps)
    log_term = torch.where(
        y_true > 0,
        y_true * torch.log(y_true.clamp(min=eps) / y_pred),
        torch.zeros_like(y_true),
    )
    return 2.0 * (y_pred - y_true + log_term).mean()


def gamma_deviance_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean Gamma deviance (masked to claims > 0):
        D = 2 · Σ[(y - ŷ)/ŷ − log(y/ŷ)] / n_claims
    """
    eps    = 1e-7
    mask   = y_true > 0
    y_pred = y_pred.clamp(min=eps)
    y_safe = y_true.clamp(min=eps)
    dev    = 2.0 * ((y_safe - y_pred) / y_pred - torch.log(y_safe / y_pred))
    n_claims = mask.float().sum().clamp(min=1.0)
    return (dev * mask.float()).sum() / n_claims


def combined_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                  target_types: list) -> torch.Tensor:
    """Loss = mean over targets of their individual deviances."""
    total = torch.tensor(0.0, device=y_true.device)
    for i, tt in enumerate(target_types):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if tt == "poisson":
            total = total + poisson_deviance_loss(yt, yp)
        else:
            total = total + gamma_deviance_loss(yt, yp)
    return total / len(target_types)


# =============================================================================
# SECTION 4 – MODEL ARCHITECTURE
# =============================================================================

class AutoTRFModel(nn.Module):
    """
    Multi-Output Transformer model for joint insurance pricing.

    Architecture (see module docstring for full description):

        1. Target-specific embeddings: for each target t, embed all covariates
           independently → h_t ∈ R^d
        2. Per-target FFN → initial scalar prediction ŷ_t^(FFN)
        3. Stack: H = [h_1, ..., h_k] ∈ R^{k × d}
        4. Add positional embedding; append CLS token: H' ∈ R^{(k+1) × d}
        5. Transformer → z_CLS ∈ R^d
        6. CLS → augmentation a ∈ R^k
        7. p_aug = σ(a + FFN_preds)
        8. Freq outputs × exposure; sev outputs as sigmoid scores

    Parameters
    ----------
    cat_vocab    : {col: n_unique_values}  — vocabulary size per cat. variable
    cont_deciles : {col: boundaries_array} — PLE bin boundaries per cont. var
    """

    def __init__(
        self,
        cat_vocab:    dict,
        cont_deciles: dict,
        target_names: list  = TARGET_NAMES,
        target_types: list  = TARGET_TYPES,
        embed_dim:    int   = EMBED_DIM,
        n_heads:      int   = N_HEADS,
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.target_names = target_names
        self.target_types = target_types
        self.cat_cols     = list(cat_vocab.keys())
        self.cont_cols    = list(cont_deciles.keys())

        k = len(target_names)
        m = len(cat_vocab) + len(cont_deciles)   # total number of features
        d = m * embed_dim                         # token dimension (per target)
        self.k = k
        self.d = d

        # ── Target-specific categorical embeddings ────────────────────────────
        # Keys: "{target_name}_{col_name}"
        self.cat_embs = nn.ModuleDict({
            f"{t}_{col}": nn.Embedding(vocab_size + 1, embed_dim)
            for t in target_names
            for col, vocab_size in cat_vocab.items()
        })

        # ── Target-specific PLE encoders + linear projections ─────────────────
        self.ple_enc = nn.ModuleDict({
            f"{t}_{col}": PiecewiseLinearEncoding(bounds.tolist())
            for t in target_names
            for col, bounds in cont_deciles.items()
        })
        self.ple_proj = nn.ModuleDict({
            f"{t}_{col}": nn.Linear(len(bounds) - 1, embed_dim)
            for t in target_names
            for col, bounds in cont_deciles.items()
        })

        # ── Per-target BatchNorm + shallow FFN → initial scalar prediction ─────
        ffn_hidden = max(8, d // 2)
        self.bn   = nn.ModuleDict({t: nn.BatchNorm1d(d) for t in target_names})
        self.ffn1 = nn.ModuleDict({t: nn.Linear(d, ffn_hidden) for t in target_names})
        self.dp   = nn.ModuleDict({t: nn.Dropout(dropout) for t in target_names})
        self.pred = nn.ModuleDict({t: nn.Linear(ffn_hidden, 1) for t in target_names})

        # ── Positional embedding: one learnable vector per target position ─────
        self.pos_emb = nn.Parameter(torch.empty(k, d))
        nn.init.uniform_(self.pos_emb)

        # ── CLS token embedding (BERT-style, Devlin et al. 2019) ─────────────
        # A global vector h_CLS is appended before the transformer.
        # After attention, h_CLS aggregates cross-target information.
        self.cls_emb = nn.Parameter(torch.empty(1, 1, d))
        nn.init.uniform_(self.cls_emb)

        # ── Transformer encoder ───────────────────────────────────────────────
        self.transformer = TransformerEncoderBlock(
            model_dim=d, n_heads=n_heads, ffn_dim=d * 2, dropout_rate=dropout
        )

        # ── CLS token → k-dimensional cross-target augmentation ──────────────
        self.cls_dense1  = nn.Linear(d, d)
        self.cls_dropout = nn.Dropout(dropout)
        self.cls_out     = nn.Linear(d, k)

    def forward(
        self,
        cat_inputs:  dict,            # {col: LongTensor (batch,)}
        cont_inputs: dict,            # {col: FloatTensor (batch,)}
        exposure:    torch.Tensor,    # (batch,) or (batch, 1)
    ) -> torch.Tensor:
        """
        Returns predicted values of shape (batch, k).
        """
        h_list    = []
        pred_list = []

        for t in self.target_names:
            parts = []

            # Categorical: entity embedding per target
            for col in self.cat_cols:
                idx = cat_inputs[col].long()
                if idx.dim() > 1:
                    idx = idx.squeeze(-1)
                emb = self.cat_embs[f"{t}_{col}"](idx)           # (batch, embed_dim)
                parts.append(emb)

            # Continuous: piecewise-linear encoding + projection per target
            for col in self.cont_cols:
                xc      = cont_inputs[col]
                ple_out = self.ple_enc[f"{t}_{col}"](xc)         # (batch, n_bins)
                proj    = F.gelu(self.ple_proj[f"{t}_{col}"](ple_out))  # (batch, embed_dim)
                parts.append(proj)

            h_t  = torch.cat(parts, dim=-1)                      # (batch, d)
            h_t  = self.bn[t](h_t)
            ffn  = F.gelu(self.ffn1[t](h_t))
            ffn  = self.dp[t](ffn)
            pred = self.pred[t](ffn)                             # (batch, 1)

            h_list.append(h_t)
            pred_list.append(pred)

        # ── Stack → H ∈ R^{batch, k, d} ──────────────────────────────────────
        H = torch.stack(h_list, dim=1)                           # (batch, k, d)

        # ── Positional embedding ──────────────────────────────────────────────
        H_pos = H + self.pos_emb.unsqueeze(0)                    # (batch, k, d)

        # ── CLS token appended → H' ∈ R^{batch, k+1, d} ─────────────────────
        batch_size = H.size(0)
        cls     = self.cls_emb.expand(batch_size, -1, -1)        # (batch, 1, d)
        H_prime = torch.cat([H_pos, cls], dim=1)                 # (batch, k+1, d)

        # ── Multi-head self-attention transformer ─────────────────────────────
        _, cls_token = self.transformer(H_prime)                 # (batch, d)

        # ── CLS token → cross-target augmentation a ∈ R^k ────────────────────
        a = F.gelu(self.cls_dense1(cls_token))
        a = self.cls_dropout(a)
        a = self.cls_out(a)                                      # (batch, k)

        # ── Combine FFN predictions with CLS augmentation ─────────────────────
        ffn_preds = torch.cat(pred_list, dim=-1)                 # (batch, k)
        p_aug     = torch.sigmoid(ffn_preds + a)                 # (batch, k)

        # ── Final per-target outputs ──────────────────────────────────────────
        exp = exposure.float()
        if exp.dim() > 1:
            exp = exp.squeeze(-1)                                # (batch,)

        outputs = []
        for i, (_, t_type) in enumerate(zip(self.target_names, self.target_types)):
            p_t = p_aug[:, i]
            if t_type == "poisson":
                out = p_t * exp          # expected count = rate × exposure
            else:
                out = p_t                # scaled severity ∈ (0,1)
            outputs.append(out.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)                        # (batch, k)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SECTION 5 – DATA PREPROCESSING
# =============================================================================

def integer_encode_cats(
    train: pd.DataFrame,
    cols: list,
) -> dict:
    """
    Build integer encoding maps from training data.
    Returns {col: {value: integer_index}} for each categorical column.
    """
    maps = {}
    for col in cols:
        uniq = sorted(train[col].unique())
        maps[col] = {v: i for i, v in enumerate(uniq)}
    return maps


def encode_split(df: pd.DataFrame, cat_maps: dict, cont_stats: dict,
                 sev_scalers: dict, target_types: list,
                 target_names: list) -> tuple[dict, np.ndarray]:
    """
    Encode a dataframe split into model inputs (X dict) and targets (y array).

    Categorical  → integer index
    Continuous   → clipped-standardised to [0, 1] using training stats
    Severity     → min-max scaled to [0, 1] using training scaler
    """
    X = {}

    # Categorical inputs
    for col, mp in cat_maps.items():
        X[f"{col}_input"] = df[col].map(mp).values.reshape(-1, 1)

    # Continuous inputs (clip to [0, 1] using training quantiles)
    for col, (q0, q1) in cont_stats.items():
        raw = df[col].clip(q0, q1).values
        scaled = (raw - q0) / (q1 - q0 + 1e-8)
        X[f"{col}_input"] = scaled.reshape(-1, 1)

    X["exposure_input"] = df["exposure"].values.reshape(-1, 1)

    # Targets
    y_cols = []
    for t_name, t_type in zip(target_names, target_types):
        guarantee = t_name.replace("freq_", "").replace("sev_", "")
        if t_type == "poisson":
            y_cols.append(df[f"num_{guarantee}"].values)
        else:                          # gamma: scaled severity
            raw_sev = df[f"sev_{guarantee}"].values
            scaled  = sev_scalers[guarantee].transform(
                raw_sev.reshape(-1, 1)
            ).flatten()
            y_cols.append(scaled)

    y = np.column_stack(y_cols).astype("float32")
    return X, y


def build_sev_scalers(train: pd.DataFrame) -> dict:
    """Fit MinMaxScaler on per-claim severity values in training data."""
    scalers = {}
    for g in ["property", "liability"]:
        mask   = train[f"num_{g}"] > 0
        scaler = MinMaxScaler()
        if mask.any():
            scaler.fit(train.loc[mask, f"sev_{g}"].values.reshape(-1, 1))
        else:
            scaler.fit(np.array([[0.0], [1.0]]))
        scalers[g] = scaler
    return scalers


def build_cont_stats(train: pd.DataFrame, cont_vars: list) -> dict:
    """Return {col: (q0, q1)} for clipping & scaling."""
    stats = {}
    for col in cont_vars:
        stats[col] = (train[col].quantile(0.01), train[col].quantile(0.99))
    return stats


def compute_deciles(train: pd.DataFrame, cont_vars: list,
                    cont_stats: dict, n_bins: int = N_BINS) -> dict:
    """Compute bin boundaries for PiecewiseLinearEncoding on [0,1]-scaled vars."""
    deciles = {}
    for col, (q0, q1) in cont_stats.items():
        raw = train[col].clip(q0, q1).values
        scaled = (raw - q0) / (q1 - q0 + 1e-8)
        deciles[col] = np.percentile(scaled, np.linspace(0, 100, n_bins + 1))
    return deciles


# =============================================================================
# SECTION 6 – METRICS
# =============================================================================

def poisson_deviance_np(y_true, y_pred):
    eps  = 1e-10
    yp   = np.maximum(y_pred, eps)
    log_ = np.where(y_true > 0,
                    y_true * np.log(np.maximum(y_true, eps) / yp),
                    0.0)
    return 2.0 * np.mean(yp - y_true + log_)


def gamma_deviance_np(y_true, y_pred, mask):
    eps  = 1e-10
    yp   = np.maximum(y_pred[mask], eps)
    yt   = np.maximum(y_true[mask], eps)
    dev  = 2.0 * ((yt - yp) / yp - np.log(yt / yp))
    return dev.mean() if len(dev) > 0 else np.nan


def rmse_np(y_true, y_pred, mask=None):
    diff = (y_true - y_pred) ** 2
    if mask is not None:
        diff = diff[mask]
    return np.sqrt(diff.mean())


# =============================================================================
# SECTION 7 – DATASET + TRAINING
# =============================================================================

class AutoInsuranceDataset(Dataset):
    """PyTorch Dataset wrapping the encoded X dict and y array."""

    def __init__(self, X: dict, y: np.ndarray, cat_cols: list, cont_cols: list):
        self.cat_data  = {
            col: torch.tensor(X[f"{col}_input"], dtype=torch.long)
            for col in cat_cols
        }
        self.cont_data = {
            col: torch.tensor(X[f"{col}_input"], dtype=torch.float32)
            for col in cont_cols
        }
        self.exposure = torch.tensor(X["exposure_input"], dtype=torch.float32)
        self.y        = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        cat  = {col: self.cat_data[col][idx]  for col in self.cat_data}
        cont = {col: self.cont_data[col][idx] for col in self.cont_data}
        return cat, cont, self.exposure[idx], self.y[idx]


def _to_device(cat, cont, exp, y_batch):
    cat  = {k: v.to(DEVICE) for k, v in cat.items()}
    cont = {k: v.to(DEVICE) for k, v in cont.items()}
    return cat, cont, exp.to(DEVICE), y_batch.to(DEVICE)


def train_model(
    model:     AutoTRFModel,
    X_train:   dict,
    y_train:   np.ndarray,
    X_val:     dict,
    y_val:     np.ndarray,
    cat_cols:  list = CAT_VARS,
    cont_cols: list = CONT_VARS,
    target_types: list = TARGET_TYPES,
) -> dict:
    """
    Train with AdamW + ReduceLROnPlateau scheduler + early stopping.
    Returns history dict with 'loss' and 'val_loss' lists.
    """
    train_ds = AutoInsuranceDataset(X_train, y_train, cat_cols, cont_cols)
    val_ds   = AutoInsuranceDataset(X_val,   y_val,   cat_cols, cont_cols)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.02,
                      betas=(0.9, 0.95))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=8,
                                  min_lr=1e-5)

    best_val_loss   = float("inf")
    best_state      = None
    patience_count  = 0
    history         = {"loss": [], "val_loss": []}

    print(f"\n  Training for up to {MAX_EPOCHS} epochs "
          f"(early stop patience={PATIENCE})…")

    for epoch in range(MAX_EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for cat, cont, exp, y_batch in train_loader:
            cat, cont, exp, y_batch = _to_device(cat, cont, exp, y_batch)
            optimizer.zero_grad()
            preds = model(cat, cont, exp)
            loss  = combined_loss(y_batch, preds, target_types)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for cat, cont, exp, y_batch in val_loader:
                cat, cont, exp, y_batch = _to_device(cat, cont, exp, y_batch)
                preds = model(cat, cont, exp)
                loss  = combined_loss(y_batch, preds, target_types)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:>4}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={lr:.2e}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    return history


def predict(
    model:     AutoTRFModel,
    X:         dict,
    cat_cols:  list = CAT_VARS,
    cont_cols: list = CONT_VARS,
) -> np.ndarray:
    """Run inference in batches; return numpy array (n, k)."""
    dummy_y = np.zeros((len(X["exposure_input"]), len(TARGET_NAMES)),
                       dtype="float32")
    ds     = AutoInsuranceDataset(X, dummy_y, cat_cols, cont_cols)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)

    model.eval()
    parts = []
    with torch.no_grad():
        for cat, cont, exp, _ in loader:
            cat  = {k: v.to(DEVICE) for k, v in cat.items()}
            cont = {k: v.to(DEVICE) for k, v in cont.items()}
            exp  = exp.to(DEVICE)
            out  = model(cat, cont, exp)
            parts.append(out.cpu().numpy())
    return np.vstack(parts)


def evaluate_model(model, X, y, sev_scalers, target_names, target_types,
                   split="test"):
    """Print Poisson deviance (freq) and RMSE + Gamma deviance (sev)."""
    preds = predict(model, X)                                    # (n, 4)

    print(f"\n  {'─' * 62}")
    print(f"  Performance on {split} set")
    print(f"  {'─' * 62}")
    print(f"  {'Target':<22} {'Poisson Dev':>12}  {'Gamma Dev':>10}  {'RMSE':>10}")
    print(f"  {'─' * 62}")

    for i, (t_name, t_type) in enumerate(zip(target_names, target_types)):
        yt = y[:, i]
        yp = preds[:, i]
        guarantee = t_name.replace("freq_", "").replace("sev_", "")

        if t_type == "poisson":
            pd_ = poisson_deviance_np(yt, yp)
            print(f"  {t_name:<22} {pd_:>12.4f}  {'—':>10}  {'—':>10}")

        else:   # gamma — inverse-scale before computing metrics
            scaler = sev_scalers[guarantee]
            yt_raw = scaler.inverse_transform(yt.reshape(-1, 1)).flatten()
            yp_raw = scaler.inverse_transform(yp.reshape(-1, 1)).flatten()
            mask   = yt > 0
            gd     = gamma_deviance_np(yt_raw, yp_raw, mask)
            rmse   = rmse_np(yt_raw, yp_raw, mask)
            print(f"  {t_name:<22} {'—':>12}  {gd:>10.4f}  {rmse:>10.1f}")

    print(f"  {'─' * 62}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 66)
    print("  AUTO INSURANCE MULTI-OUTPUT TRANSFORMER (PyTorch)")
    print("=" * 66)
    print(f"  Device: {DEVICE}")

    # ── 1. Simulate data ──────────────────────────────────────────────────────
    print("\n[1] Simulating auto insurance portfolio…")
    df = simulate_auto_portfolio(N_POLICIES)
    print(f"  N = {len(df):,}  |  train={df.group.eq('train').sum():,}  "
          f"val={df.group.eq('valid').sum():,}  "
          f"test={df.group.eq('test').sum():,}")
    print_kpis(df)

    train_df = df[df.group == "train"].reset_index(drop=True)
    val_df   = df[df.group == "valid"].reset_index(drop=True)
    test_df  = df[df.group == "test" ].reset_index(drop=True)

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    print("\n[2] Preprocessing…")
    cat_maps     = integer_encode_cats(train_df, CAT_VARS)
    cont_stats   = build_cont_stats(train_df, CONT_VARS)
    cont_deciles = compute_deciles(train_df, CONT_VARS, cont_stats, N_BINS)
    sev_scalers  = build_sev_scalers(train_df)

    cat_vocab = {col: len(mp) for col, mp in cat_maps.items()}
    print(f"  Cat vocab sizes: { {c: v for c, v in cat_vocab.items()} }")
    print(f"  Cont var ranges: "
          f"{ {c: (round(v[0],1), round(v[1],1)) for c, v in cont_stats.items()} }")

    def enc(split_df):
        return encode_split(
            split_df, cat_maps, cont_stats, sev_scalers,
            TARGET_TYPES, TARGET_NAMES
        )

    X_train, y_train = enc(train_df)
    X_val,   y_val   = enc(val_df)
    X_test,  y_test  = enc(test_df)

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[3] Building TRF model…")
    model = AutoTRFModel(
        cat_vocab    = cat_vocab,
        cont_deciles = cont_deciles,
        embed_dim    = EMBED_DIM,
        n_heads      = N_HEADS,
        dropout      = DROPOUT,
    ).to(DEVICE)

    total_params = model.count_parameters()
    print(f"  Total trainable parameters: {total_params:,}")

    k = len(TARGET_NAMES)
    m = len(CAT_VARS) + len(CONT_VARS)
    d = m * EMBED_DIM
    print(f"  k={k} targets | m={m} features | embed_dim={EMBED_DIM} | "
          f"token_dim d={d} | n_heads={N_HEADS}")
    print(f"  Transformer sequence: {k+1} tokens ({k} target tokens + 1 CLS)")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[4] Training…")
    history = train_model(model, X_train, y_train, X_val, y_val)
    print(f"\n  Best val_loss: {min(history['val_loss']):.5f}  "
          f"(stopped at epoch {len(history['loss'])})")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5] Evaluation…")
    evaluate_model(model, X_train, y_train, sev_scalers,
                   TARGET_NAMES, TARGET_TYPES, "train")
    evaluate_model(model, X_test,  y_test,  sev_scalers,
                   TARGET_NAMES, TARGET_TYPES, "test")

    print("\n  Done.")
    return model, df


if __name__ == "__main__":
    main()
