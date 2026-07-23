"""
=============================================================================
IMPROVED AUTO INSURANCE DATA SIMULATION ENGINE
=============================================================================

Design
------
Two-stage simulation (preserving simulate_data.py logic):

  Stage 1  – Draw policy-level accident counts from a Poisson process
              whose rate is a multiplicative function of all covariates
              plus a log-normal latent frailty factor Z (shared shock).

  Stage 2  – For each accident, independently determine whether PD and BI
              claims arise; draw claim severities via a bivariate Gaussian
              copula (within-accident PD/BI correlation ρ = 0.55).
              Then aggregate counts and costs back to policy level.

What changed vs. the original simulate_data.py
-----------------------------------------------
Preserved  : Beta-distributed age, exponential vehicle age with
             heterogeneous mean, vehicle weight categories, shared
             latent frailty Z, two-stage accident → claim structure,
             bivariate Gaussian copula for PD/BI severity.

Added      : territory       Urban / Suburban / Rural
             annual_mileage  continuous; territory-dependent log-normal
             prior_claims_3yr  0 / 1 / 2+  experience-rating factor
             vehicle_value   continuous; log-normal, age-decreasing
             safety_score    [0,1] proxy for ADAS / crash-avoidance tech
             credit_tier     A / B / C / D  behavioural proxy
             vehicle_use     Commute / Pleasure / Commercial
             marital_status  Married / Single
             exposure        fraction of year insured [0.25, 1.0]

Improved   : Vectorised claim loop (replaces Python for-loop → 50-100×
             faster for large N), richer multiplicative frequency model,
             territory- and vehicle-value-driven severity calibration,
             safety-score effect on BI injury cost, 70/10/20 split.

Output columns compatible with auto_insurance_transformer.py:
  Frequency targets : num_property  (PD claim count per policy)
                      num_liability (BI claim count per policy)
  Severity targets  : sev_property  (total PD cost,  0 if no claim)
                      sev_liability (total BI cost,   0 if no claim)
  Key covariates    : driver_age, vehicle_age, vehicle_weight,
                      territory, annual_mileage, prior_claims_3yr,
                      vehicle_value, safety_score, credit_tier,
                      vehicle_use, marital_status, exposure
  Latent variable   : latent_z (frailty; not observable in real data)
  Split             : group  ∈ {train, valid, test}

Dependency structure
--------------------
Within a single accident, PD and BI severities are correlated via a
bivariate Gaussian copula (ρ_sev = 0.55): a high-energy crash damages
both the vehicle heavily AND injures occupants severely.

Across policies, unobserved heterogeneity (driving style, attention,
risk appetite) is captured by Z ~ N(0,1).  The frailty term exp(σ_Z · Z)
creates overdispersion in claim counts beyond the Poisson baseline.

Factor structure  (multiplicative on accident rate)
---------------------------------------------------
Covariate         Values              Relative rate
-----------       -----------------   ----------------------------------
driver_age        <25                 +70 %  (inexperience)
                  25–64               baseline
                  65+                 +35 %  (slowing reflexes)
territory         Urban               +50 %  (density, road complexity)
                  Suburban            baseline
                  Rural               −30 %  (open roads, less traffic)
annual_mileage    10 000 km           power-law mileage^0.55 / ref^0.55
prior_claims_3yr  0                   −10 %
                  1                   +55 %
                  2+                  +140 %
credit_tier       A                   −18 %
                  B                   baseline
                  C                   +22 %
                  D                   +58 %
vehicle_use       Commercial          +45 %
                  Commute             +12 %
                  Pleasure            −12 %
marital_status    Single              +10 %
                  Married             −10 %
vehicle_weight    Heavy               +15 %  (blind spots, stopping distance)
                  Medium              baseline
                  Light               −10 %
latent_Z          N(0,1)              exp(0.30 · Z)  → overdispersion
"""

# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma as gamma_dist

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED        = 42
N_POLICIES  = 500_000

# Copula correlation: within-accident PD ↔ BI severity (same crash energy)
RHO_SEV     = 0.55

# Gamma shape for PD severity (controls dispersion around mean)
PD_ALPHA    = 2.5

# Log-normal sigma for BI severity
BI_SIGMA    = 1.15

# Frailty standard deviation on log scale
FRAILTY_STD = 0.30

# Base accident frequency (per policy-year)
BASE_FREQ   = 0.060

# =============================================================================
# SECTION 1 – COVARIATE SIMULATION
# =============================================================================

def _draw_covariates(n: int, rng: np.random.Generator) -> dict:
    """
    Draw all raw covariates for n policies.
    Returns a dict of numpy arrays, all length n.
    """
    # ── Driver age: Beta(2,5) on [18, 93] ────────────────────────────────────
    # Right-skewed toward younger ages; more realistic than uniform.
    driver_age = np.round(18 + rng.beta(2, 5, n) * 75).astype(int)

    # ── Vehicle age: Exponential with heterogeneous mean ─────────────────────
    # Each policy has its own expected vehicle age drawn from U[3.5, 6.5].
    # Capped at 30 years. Produces realistic mixture of new and old vehicles.
    lam_veh    = rng.uniform(3.5, 6.5, n)
    vehicle_age = np.minimum(30, np.round(rng.exponential(lam_veh)).astype(int))

    # ── Vehicle weight ────────────────────────────────────────────────────────
    vehicle_weight = rng.choice(
        ["Light", "Medium", "Heavy"], p=[0.50, 0.35, 0.15], size=n
    )

    # ── Territory ─────────────────────────────────────────────────────────────
    # Urban: high density, complex intersections → more accidents, higher costs.
    # Rural: open roads, lower frequency but higher severity when crash occurs.
    territory = rng.choice(
        ["Urban", "Suburban", "Rural"], p=[0.35, 0.45, 0.20], size=n
    )

    # ── Annual mileage: territory-dependent log-normal ────────────────────────
    # Urban commuters drive more; rural drivers drive less (but in different
    # conditions).  Values in km/year; mean ~14 000 km (Suburban reference).
    mu_mi  = np.where(territory == "Urban",    9.70,
              np.where(territory == "Suburban", 9.50, 9.30))   # ln(km)
    sig_mi = np.where(territory == "Urban",    0.45,
              np.where(territory == "Suburban", 0.50, 0.55))
    annual_mileage = np.round(
        np.exp(mu_mi + sig_mi * rng.standard_normal(n))
    ).astype(int)
    annual_mileage = np.clip(annual_mileage, 1_000, 120_000)

    # ── Prior claims in the last 3 years ─────────────────────────────────────
    # Strong forward-looking predictor: past behaviour predicts future risk.
    prior_claims_3yr = rng.choice([0, 1, 2], p=[0.80, 0.15, 0.05], size=n)

    # ── Vehicle value (€): log-normal, decreasing with vehicle age ───────────
    # Represents replacement/repair cost ceiling for PD severity.
    vv_mu  = 10.50 - 0.065 * vehicle_age   # ln(€); new car ≈ €36 000
    vv_sig = 0.55
    vehicle_value = np.round(
        np.exp(vv_mu + vv_sig * rng.standard_normal(n)), -2
    )
    vehicle_value = np.clip(vehicle_value, 500, 250_000).astype(float)

    # ── Safety score: [0, 1] – proxy for ADAS / crash-avoidance tech ─────────
    # Newer vehicles have more automated braking, lane-keeping, etc.
    # Logistic of (vehicle_age) centred so age≈0 → score≈0.85, age≈15 → 0.35.
    safety_score = 1.0 / (1.0 + np.exp(0.30 * vehicle_age - 2.8))
    safety_score += 0.05 * rng.standard_normal(n)
    safety_score  = np.clip(safety_score, 0.0, 1.0).round(3)

    # ── Credit tier: A/B/C/D – behavioural proxy ─────────────────────────────
    # Correlated with risk-taking behaviour in actuarial literature.
    credit_tier = rng.choice(
        ["A", "B", "C", "D"], p=[0.30, 0.40, 0.20, 0.10], size=n
    )

    # ── Marital status ────────────────────────────────────────────────────────
    marital_status = rng.choice(["Married", "Single"], p=[0.60, 0.40], size=n)

    # ── Vehicle use ───────────────────────────────────────────────────────────
    # Commercial vehicles (delivery, taxis) accumulate more mileage in
    # high-risk conditions.
    vehicle_use = rng.choice(
        ["Commute", "Pleasure", "Commercial"], p=[0.50, 0.35, 0.15], size=n
    )

    # ── Exposure: fraction of year insured ───────────────────────────────────
    exposure = np.round(rng.uniform(0.25, 1.0, n), 2)

    # ── Latent frailty: unobserved driving behaviour / risk propensity ────────
    # Z ~ N(0,1); enters the model as exp(FRAILTY_STD * Z).
    # Creates overdispersion in claim counts (Negative-Binomial-like).
    latent_z = rng.standard_normal(n)

    return dict(
        driver_age       = driver_age,
        vehicle_age      = vehicle_age,
        vehicle_weight   = vehicle_weight,
        territory        = territory,
        annual_mileage   = annual_mileage,
        prior_claims_3yr = prior_claims_3yr,
        vehicle_value    = vehicle_value,
        safety_score     = safety_score,
        credit_tier      = credit_tier,
        marital_status   = marital_status,
        vehicle_use      = vehicle_use,
        exposure         = exposure,
        latent_z         = latent_z,
    )


# =============================================================================
# SECTION 2 – FREQUENCY MODEL
# =============================================================================

def _compute_accident_lambda(cov: dict) -> np.ndarray:
    """
    Compute per-policy expected accident count (pre-exposure).

    All factors are multiplicative on the base rate.  The latent frailty
    exp(FRAILTY_STD · Z) adds Gamma-mixture overdispersion.

    Returns lambda_i · exposure_i (ready for Poisson draw).
    """
    age  = cov["driver_age"]
    terr = cov["territory"]
    mi   = cov["annual_mileage"]
    pc   = cov["prior_claims_3yr"]
    cr   = cov["credit_tier"]
    use  = cov["vehicle_use"]
    mar  = cov["marital_status"]
    wt   = cov["vehicle_weight"]
    Z    = cov["latent_z"]
    exp_ = cov["exposure"]

    # Age: dual-bump U-shape (young + elderly peaks)
    age_factor = (
        1.00
        + 0.70 * (age < 25)
        + 0.20 * (age >= 25) * (age < 30) * (30 - age) / 5   # taper off
        + 0.35 * (age > 65)
    )

    # Territory: road density and complexity
    terr_factor = np.where(terr == "Urban",    1.50,
                  np.where(terr == "Suburban",  1.00, 0.70))

    # Annual mileage: power law relative to 14 000 km reference
    mileage_factor = np.power(np.maximum(mi, 1) / 14_000, 0.55)

    # Prior claims: experience-rated credibility factor
    prior_factor = np.where(pc == 0, 0.90,
                   np.where(pc == 1, 1.55, 2.40))

    # Credit tier: behavioural proxy
    credit_factor = np.where(cr == "A", 0.82,
                    np.where(cr == "B", 1.00,
                    np.where(cr == "C", 1.22, 1.58)))

    # Vehicle use: operational intensity
    use_factor = np.where(use == "Commercial", 1.45,
                 np.where(use == "Commute",    1.12, 0.88))

    # Marital status: small but empirically consistent effect
    mar_factor = np.where(mar == "Married", 0.90, 1.10)

    # Vehicle weight: heavier vehicles have larger blind spots, longer stopping
    wt_factor  = np.where(wt == "Heavy",  1.15,
                 np.where(wt == "Medium", 1.00, 0.90))

    # Latent frailty: log-normal overdispersion
    frailty = np.exp(FRAILTY_STD * Z)

    lambda_rate = (
        BASE_FREQ
        * age_factor
        * terr_factor
        * mileage_factor
        * prior_factor
        * credit_factor
        * use_factor
        * mar_factor
        * wt_factor
        * frailty
    )

    return lambda_rate * exp_   # Poisson parameter = rate × exposure


# =============================================================================
# SECTION 3 – CLAIM-LEVEL SIMULATION (VECTORISED)
# =============================================================================

def _simulate_claims_vectorised(
    cov:          dict,
    num_accidents: np.ndarray,
    rng:          np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2: for each accident, determine PD/BI occurrence and severity.

    Instead of looping over policies, we:
      (a) build a flat array of all accidents via np.repeat,
      (b) draw claim indicators and copula-correlated severities in bulk,
      (c) aggregate back to policy level with np.bincount.

    Returns (pd_count, bi_count, pd_total_cost, bi_total_cost),
    each an array of shape (n_policies,).
    """
    n              = len(num_accidents)
    total_acc      = int(num_accidents.sum())

    # Trivial case: no accidents at all
    if total_acc == 0:
        zeros = np.zeros(n)
        return zeros, zeros, zeros, zeros

    # Map each accident to its parent policy
    pol_idx = np.repeat(np.arange(n), num_accidents)   # (total_acc,)

    # ── PD indicator: ~95% of accidents produce a PD claim ───────────────────
    has_pd = rng.binomial(1, 0.95, total_acc).astype(bool)

    # ── BI probability: depends on vehicle weight and territory ──────────────
    # Heavy vehicles are more likely to seriously injure the other party.
    # Urban environments have more vulnerable road users (pedestrians, cyclists).
    bi_base = np.where(cov["vehicle_weight"][pol_idx] == "Heavy",  0.38,
               np.where(cov["vehicle_weight"][pol_idx] == "Medium", 0.26, 0.16))
    bi_terr = np.where(cov["territory"][pol_idx] == "Urban",    1.30,
               np.where(cov["territory"][pol_idx] == "Suburban", 1.00, 0.75))
    # Safety features reduce BI probability (automatic braking, airbags)
    bi_safety = 1.0 - 0.20 * cov["safety_score"][pol_idx]

    bi_prob = np.clip(bi_base * bi_terr * bi_safety, 0.0, 1.0)
    has_bi  = rng.binomial(1, bi_prob).astype(bool)

    # ── Bivariate Gaussian copula for PD/BI severity ─────────────────────────
    # Within the same accident, the energy of the crash drives both the
    # vehicle damage (PD) and the occupant/pedestrian injury (BI) jointly.
    # We model this dependency with a Gaussian copula (ρ = RHO_SEV = 0.55).
    cov_mat = np.array([[1.0, RHO_SEV], [RHO_SEV, 1.0]])
    z_cop   = rng.multivariate_normal([0.0, 0.0], cov_mat, total_acc)
    u_cop   = norm.cdf(z_cop)   # probability-integral-transform → Uniform(0,1)

    # ── PD severity: Gamma(α=2.5, scale=f(vehicle_value, territory, safety)) ──
    # Vehicle value sets the expected repair cost ceiling.
    # Urban labour and parts cost premiums are captured by terr_sev.
    # Safety score slightly reduces damage (crumple zones absorb impact).
    terr_sev   = np.where(cov["territory"][pol_idx] == "Urban",    1.35,
                  np.where(cov["territory"][pol_idx] == "Suburban",  1.00, 0.78))
    safety_red = 1.0 - 0.12 * cov["safety_score"][pol_idx]

    # Scale calibrated so mean PD ≈ 20% of vehicle value (Gamma mean = α × scale).
    # pd_scale = vv * 0.08  →  mean = 2.5 * 0.08 * vv = 0.20 * vv
    # Examples: €5k car → mean ~€1 000 ; €20k car → mean ~€4 000 ; €60k → ~€12 000
    pd_scale = cov["vehicle_value"][pol_idx] * 0.08 * terr_sev * safety_red
    pd_scale = np.clip(pd_scale, 100.0, 80_000.0)

    pd_severity = gamma_dist.ppf(
        np.clip(u_cop[:, 0], 1e-6, 1 - 1e-6),
        a=PD_ALPHA, scale=pd_scale
    )
    pd_severity = np.where(has_pd, pd_severity, 0.0)

    # ── BI severity: log-normal with covariate-driven mean ───────────────────
    # The log-mean is a linear combination of injury-cost drivers:
    #   - elderly victims: higher medical cost, longer recovery
    #   - young at-fault: more severe injury patterns
    #   - urban: attorney involvement, higher medical billing
    #   - heavy vehicles: more kinetic energy transferred to victim
    #   - safety score: active braking reduces collision speed → lower injury
    age_bi = cov["driver_age"][pol_idx]
    bi_log_mu = (
        8.60                                                              # ln(€) base
        + 0.30 * (age_bi > 65).astype(float)                             # elderly
        + 0.18 * (age_bi < 25).astype(float)                             # young
        + 0.35 * (cov["territory"][pol_idx] == "Urban").astype(float)    # urban costs
        + 0.10 * (cov["territory"][pol_idx] == "Rural").astype(float)    # rural: speed
        + 0.18 * (cov["vehicle_weight"][pol_idx] == "Heavy").astype(float)  # impact mass
        - 0.25 * cov["safety_score"][pol_idx]                            # ADAS mitigation
    )
    bi_severity = np.exp(
        bi_log_mu + BI_SIGMA * norm.ppf(np.clip(u_cop[:, 1], 1e-6, 1 - 1e-6))
    )
    bi_severity = np.where(has_bi, bi_severity, 0.0)

    # ── Aggregate to policy level ─────────────────────────────────────────────
    pd_count      = np.bincount(pol_idx, weights=has_pd.astype(float), minlength=n)
    bi_count      = np.bincount(pol_idx, weights=has_bi.astype(float), minlength=n)
    pd_total_cost = np.bincount(pol_idx, weights=pd_severity,          minlength=n)
    bi_total_cost = np.bincount(pol_idx, weights=bi_severity,          minlength=n)

    return pd_count, bi_count, pd_total_cost, bi_total_cost


# =============================================================================
# SECTION 4 – MAIN SIMULATION FUNCTION
# =============================================================================

def simulate_auto_data(
    num_records:  int = N_POLICIES,
    random_seed:  int = SEED,
    add_split:    bool = True,
    split_ratios: tuple = (0.70, 0.10, 0.20),
) -> pd.DataFrame:
    """
    Simulate an auto insurance portfolio.

    Parameters
    ----------
    num_records   : number of policies to generate
    random_seed   : reproducibility seed
    add_split     : if True, append a 'group' column (train/valid/test)
    split_ratios  : (train, valid, test) fractions; must sum to 1

    Returns
    -------
    pd.DataFrame with one row per policy and the following columns:

    Covariates
        Policy_ID, Driver_Age, Vehicle_Age, Vehicle_Weight,
        Territory, Annual_Mileage, Prior_Claims_3yr,
        Vehicle_Value, Safety_Score, Credit_Tier,
        Marital_Status, Vehicle_Use, Exposure, Latent_Z

    Targets  (compatible with auto_insurance_transformer.py)
        Total_Accidents   – total accident count per policy-year
        num_property      – PD claim count  (property damage)
        num_liability     – BI claim count  (bodily injury / liability)
        sev_property      – total PD cost   (€, 0 if no PD claim)
        sev_liability     – total BI cost   (€, 0 if no BI claim)
        Total_Pure_Premium – PD + BI total cost

    Split
        group  ∈ {train, valid, test}
    """
    rng = np.random.default_rng(random_seed)

    # ── Stage 0: Draw covariates ──────────────────────────────────────────────
    cov = _draw_covariates(num_records, rng)

    # ── Stage 1: Accident counts ──────────────────────────────────────────────
    accident_lambda = _compute_accident_lambda(cov)
    num_accidents   = rng.poisson(accident_lambda).astype(int)

    # ── Stage 2: Per-accident PD/BI claim simulation ──────────────────────────
    pd_count, bi_count, pd_cost, bi_cost = _simulate_claims_vectorised(
        cov, num_accidents, rng
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "Policy_ID":         np.arange(1, num_records + 1),
        # Existing covariates (preserve original naming)
        "Driver_Age":        cov["driver_age"],
        "Vehicle_Age":       cov["vehicle_age"],
        "Vehicle_Weight":    cov["vehicle_weight"],
        # New covariates
        "Territory":         cov["territory"],
        "Annual_Mileage":    cov["annual_mileage"],
        "Prior_Claims_3yr":  cov["prior_claims_3yr"],
        "Vehicle_Value":     cov["vehicle_value"],
        "Safety_Score":      cov["safety_score"],
        "Credit_Tier":       cov["credit_tier"],
        "Marital_Status":    cov["marital_status"],
        "Vehicle_Use":       cov["vehicle_use"],
        "Exposure":          cov["exposure"],
        # Latent factor (not observable in practice; useful for diagnostics)
        "Latent_Z":          np.round(cov["latent_z"], 4),
        # Targets – naming compatible with auto_insurance_transformer.py
        "Total_Accidents":   num_accidents,
        "num_property":      pd_count.astype(int),
        "num_liability":     bi_count.astype(int),
        "sev_property":      np.round(pd_cost, 2),
        "sev_liability":     np.round(bi_cost, 2),
        "Total_Pure_Premium": np.round(pd_cost + bi_cost, 2),
    })

    # ── Train / valid / test split ────────────────────────────────────────────
    if add_split:
        assert abs(sum(split_ratios) - 1.0) < 1e-9, "split_ratios must sum to 1"
        tr, va, _ = split_ratios
        perm   = rng.permutation(num_records)
        groups = np.empty(num_records, dtype="U8")
        groups[perm[:int(tr * num_records)]]                       = "train"
        groups[perm[int(tr * num_records): int((tr + va) * num_records)]] = "valid"
        groups[perm[int((tr + va) * num_records):]]                = "test"
        df["group"] = groups

    return df


# =============================================================================
# SECTION 5 – DIAGNOSTICS
# =============================================================================

def print_diagnostics(df: pd.DataFrame) -> None:
    """
    Print an actuarial summary of the simulated portfolio.
    Useful for sanity-checking the factor structure and calibration.
    """
    n   = len(df)
    exp = df["Exposure"].sum()

    print("\n" + "=" * 70)
    print("  PORTFOLIO DIAGNOSTICS")
    print("=" * 70)
    print(f"  Policies          : {n:>10,}")
    print(f"  Total exposure    : {exp:>10,.0f} policy-years")
    print()

    # ── Overall KPIs ─────────────────────────────────────────────────────────
    for col, label in [("num_property",  "PD (property)"),
                        ("num_liability", "BI (liability)")]:
        freq   = df[col].sum() / exp
        has_cl = df[col] > 0
        sev_col = col.replace("num_", "sev_")
        avg_sev = df.loc[has_cl, sev_col].mean() if has_cl.any() else 0.0
        n_cl    = has_cl.sum()
        print(f"  {label:<18}  freq={freq:7.4%}  "
              f"mean_sev={avg_sev:>10,.0f}  n_claimants={n_cl:>7,}")

    acc_rate = df["Total_Accidents"].sum() / exp
    print(f"\n  Accident rate     : {acc_rate:.4%} per policy-year")
    print(f"  Mean pure premium : {df['Total_Pure_Premium'].mean():>10,.0f}")
    print(f"  PD/BI co-claims   : "
          f"{((df['num_property'] > 0) & (df['num_liability'] > 0)).mean():.2%} "
          f"of policies with at least one claim")

    # ── Frequency by territory ────────────────────────────────────────────────
    print("\n  Accident frequency by territory:")
    for t in ["Urban", "Suburban", "Rural"]:
        sub  = df[df["Territory"] == t]
        freq = sub["Total_Accidents"].sum() / sub["Exposure"].sum()
        print(f"    {t:<10}  {freq:.4%}  (n={len(sub):>7,})")

    # ── Frequency by prior claims ─────────────────────────────────────────────
    print("\n  Accident frequency by prior_claims_3yr:")
    for pc in [0, 1, 2]:
        sub  = df[df["Prior_Claims_3yr"] == pc]
        freq = sub["Total_Accidents"].sum() / sub["Exposure"].sum()
        print(f"    Prior={pc}    {freq:.4%}  (n={len(sub):>7,})")

    # ── PD severity by vehicle value quintile ─────────────────────────────────
    print("\n  Mean PD severity by vehicle value quintile (claimants only):")
    claimants  = df[df["num_property"] > 0].copy()
    if len(claimants) > 0:
        claimants["vv_q"] = pd.qcut(claimants["Vehicle_Value"], 5,
                                     labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for q, grp in claimants.groupby("vv_q"):
            print(f"    {q}  mean_sev={grp['sev_property'].mean():>10,.0f}  "
                  f"(n={len(grp):>6,})")

    # ── BI severity by territory ──────────────────────────────────────────────
    print("\n  Mean BI severity by territory (claimants only):")
    bi_cl = df[df["num_liability"] > 0]
    for t in ["Urban", "Suburban", "Rural"]:
        sub = bi_cl[bi_cl["Territory"] == t]
        if len(sub) > 0:
            print(f"    {t:<10}  {sub['sev_liability'].mean():>10,.0f}  "
                  f"(n={len(sub):>6,})")

    # ── Cross-guarantee dependence ─────────────────────────────────────────────
    pd_ct  = df["num_property"].values.astype(float)
    bi_ct  = df["num_liability"].values.astype(float)
    rho_ct = np.corrcoef(pd_ct, bi_ct)[0, 1]
    print(f"\n  Observed count correlation PD↔BI : {rho_ct:.3f}  "
          f"(copula-induced through shared accidents)")

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Simulating {N_POLICIES:,} policies …")
    df = simulate_auto_data(N_POLICIES, SEED)

    print_diagnostics(df)

    out_path = "sim_auto_data.csv"
    df.to_csv(out_path, index=False)
    print()
    print("Saved to", out_path, "(%d rows x %d cols)" % (df.shape[0], df.shape[1]))
