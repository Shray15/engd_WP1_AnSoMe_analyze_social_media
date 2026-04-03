"""
Run Multinomial Logistic Regression with each cluster as reference.
Three model types:
  1. posts_only   – post features only
  2. ha_only      – HA features only
  3. combined     – post + HA features

Saves significant results (p < 0.05) and full summary per reference cluster.
"""

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Data\KMeans_filtered_with_HA_feature_levels_top50.csv"
OUT_BASE  = r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\Analysis_pipeline\mlogit_post_ha_combined_2classes"

# ─── Labels ───────────────────────────────────────────────────────────────────
DISCOURSE_MAP = {
    0: "On-topic_Feedbacks",
    1: "On-topic_Criticism",
    2: "On-topic_Praise",
    3: "Off-topic_Complaints",
    4: "Content_Sharing",
    5: "Information_Seeking",
}
CLUSTERS = list(DISCOURSE_MAP.keys())  # [0,1,2,3,4,5]

# ─── Load & prepare data ──────────────────────────────────────────────────────
print("Loading data …")
ha_df = pd.read_csv(DATA_PATH)
df_model = ha_df.copy()
df_model["Post"] = df_model["Post"].astype(str)

# Post features
df_model["post_length"]       = df_model["Post"].apply(lambda x: len(x.split()))
bins   = [0, 50, 150, np.inf]
labels = ["Very_Short", "Short", "Medium_Long"]
df_model["length_category"]   = pd.cut(df_model["post_length"], bins=bins, labels=labels,
                                        right=True, include_lowest=True)
df_model["unique_words"]      = df_model["Post"].apply(lambda x: len(set(x.split())))
df_model["unique_words_ratio"] = (df_model["unique_words"] / df_model["post_length"] * 100).round(2).fillna(0)
q50 = df_model["unique_words_ratio"].quantile(0.50)
df_model["unique_words_cat"]  = df_model["unique_words_ratio"].apply(
    lambda x: "Low" if x < q50 else "High"
)
df_model["has_questions"] = df_model["Post"].str.contains(r"\?", regex=True).astype(int)
df_model["has_urls"]      = df_model["Post"].str.contains(r"http\S+", regex=True).astype(int)

# HA features (q33 threshold; two vars are inverted: low value = good outcome)
HA_FEATURES = [
    #"Operation expense/unit",
    #"Maintainance cost/unit",
    "Total Housing",
    "Affordability(Rent/month)",
    "Tenants Left",
]

# For inverted vars: low raw value = good, so label it "High"; maps to a renamed column
INVERTED_VARS = {
    "Affordability(Rent/month)": "Affordability_Level",    # low rent -> High affordability
    "Tenants Left":              "Tenant_Satisfaction_Level",  # low churn -> High satisfaction
}

HA_LEVEL_FEATS = []
for feat in HA_FEATURES:
    q50_ha = df_model[feat].quantile(0.50)
    if feat in INVERTED_VARS:
        level_col = INVERTED_VARS[feat]
        df_model[level_col] = pd.cut(
            df_model[feat], bins=[-np.inf, q50_ha, np.inf],
            labels=["High", "Low"]  # intentionally reversed
        )
    else:
        level_col = f"{feat}_Level"
        df_model[level_col] = pd.cut(
            df_model[feat], bins=[-np.inf, q50_ha, np.inf],
            labels=["Low", "High"]
        )
    HA_LEVEL_FEATS.append(level_col)

# Reference category per HA level column:
#   inverted vars -> drop "Low" (best outcome = reference)
#   normal vars   -> drop "Low"  (worst/baseline = reference)
HA_REF_SUFFIX = {col: "Low" if col in INVERTED_VARS.values() else "Low"
                 for col in HA_LEVEL_FEATS}

# Deduplicate to unique posts
df_posts = df_model.drop_duplicates(subset="Post").copy()
print(f"Unique posts: {len(df_posts)}")


# ─── Helper: build feature matrix ─────────────────────────────────────────────
def build_X(df, use_post=True, use_ha=True):
    parts = []

    if use_post:
        # unique_words_cat  (ref = Low)
        uw = pd.get_dummies(df["unique_words_cat"], prefix="uw").astype(float)
        if "uw_Low" in uw.columns:
            uw = uw.drop(columns=["uw_Low"])
        parts.append(uw)

        # length_category  (ref = Very_Short)
        lc = pd.get_dummies(df["length_category"], prefix="length").astype(float)
        ref_col = [c for c in lc.columns if "Very" in c]
        if ref_col:
            lc = lc.drop(columns=ref_col)
        parts.append(lc)

        # binary features
        parts.append(df[["has_questions", "has_urls"]].astype(float))

    if use_ha:
        for feat in HA_LEVEL_FEATS:
            safe = feat.replace("(", "").replace(")", "").replace("/", "_").replace(" ", "_")
            d = pd.get_dummies(df[feat], prefix=safe).astype(float)
            ref_suffix = HA_REF_SUFFIX[feat]
            ref_col = [c for c in d.columns if c.endswith(f"_{ref_suffix}")]
            if ref_col:
                d = d.drop(columns=ref_col)
            parts.append(d)

    return pd.concat(parts, axis=1)


# ─── Helper: fit model & extract results ──────────────────────────────────────
def fit_model(df, ref_cluster, use_post, use_ha):
    other_clusters = [c for c in CLUSTERS if c != ref_cluster]
    cluster_order  = [ref_cluster] + other_clusters
    class_names    = [DISCOURSE_MAP[c] for c in cluster_order]

    y = df["Cluster_KMeans"].map({c: i for i, c in enumerate(cluster_order)}).values
    X = build_X(df, use_post=use_post, use_ha=use_ha)
    X_final = sm.add_constant(X)

    model   = sm.MNLogit(y, X_final)
    results = model.fit(disp=False, maxiter=500)

    feature_names = list(X.columns)
    records = []
    for cls_idx, cls_name in enumerate(class_names[1:]):
        cls_num = cluster_order[cls_idx + 1]
        coeffs  = results.params.values[1:, cls_idx]
        pvals   = results.pvalues.values[1:, cls_idx]
        se      = results.bse.values[1:, cls_idx]
        for feat, coef, pv, std in zip(feature_names, coeffs, pvals, se):
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "." if pv < 0.1 else ""
            records.append({
                "Reference_Cluster"   : ref_cluster,
                "Reference_Label"     : DISCOURSE_MAP[ref_cluster],
                "Discourse_Type"      : cls_name,
                "Cluster"             : cls_num,
                "Feature"             : feat,
                "Coef"                : round(coef, 4),
                "SE"                  : round(std, 4),
                "p_value"             : round(pv, 4),
                "Sig"                 : sig,
                "Significant"         : pv < 0.05,
            })

    return pd.DataFrame(records), results, class_names


# ─── Helper: save results ─────────────────────────────────────────────────────
def save_results(results_df, fit_results, class_names, ref_cluster, folder):
    os.makedirs(folder, exist_ok=True)

    # significant rows CSV
    sig_df = results_df[results_df["Significant"]].sort_values(["Discourse_Type", "p_value"])
    sig_df.to_csv(os.path.join(folder, "significant_results.csv"), index=False)

    # full results CSV
    results_df.to_csv(os.path.join(folder, "full_results.csv"), index=False)

    # text summary
    ref_label = DISCOURSE_MAP[ref_cluster]
    summary_path = os.path.join(folder, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Reference class : {ref_label} (Cluster {ref_cluster})\n")
        f.write(f"N posts         : {int(fit_results.nobs)}\n")
        f.write(f"Pseudo R2       : {fit_results.prsquared:.4f}\n")
        f.write(f"AIC             : {fit_results.aic:.2f}\n")
        f.write(f"BIC             : {fit_results.bic:.2f}\n")
        f.write(f"Log-Likelihood  : {fit_results.llf:.2f}\n")
        f.write(f"Converged       : {fit_results.mle_retvals.get('converged', 'N/A')}\n\n")

        n_sig = results_df["Significant"].sum()
        n_tot = len(results_df)
        f.write(f"Significant     : {n_sig}/{n_tot} ({n_sig/n_tot*100:.1f}%)\n\n")

        f.write("=" * 72 + "\n")
        f.write("SIGNIFICANT EFFECTS (p < 0.05)\n")
        f.write("=" * 72 + "\n")
        hdr = f"{'Discourse Type':<30} {'Feature':<30} {'Coef':>8} {'p':>7}  Sig\n"
        f.write(hdr)
        f.write("-" * 80 + "\n")
        for _, row in sig_df.iterrows():
            f.write(
                f"{row['Discourse_Type']:<30} {row['Feature']:<30} "
                f"{row['Coef']:>8.4f} {row['p_value']:>7.4f}  {row['Sig']}\n"
            )


# ─── Run all models ───────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    ("posts_only", True,  False),
    ("ha_only",    False, True),
    ("combined",   True,  True),
]

all_significant = {name: [] for name, _, _ in MODEL_CONFIGS}

for model_name, use_post, use_ha in MODEL_CONFIGS:
    model_dir = os.path.join(OUT_BASE, model_name)
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*60}")

    for ref_cluster in CLUSTERS:
        ref_label = DISCOURSE_MAP[ref_cluster]
        folder_name = f"ref_{ref_cluster}_{ref_label}"
        out_folder  = os.path.join(model_dir, folder_name)

        print(f"  Reference: Cluster {ref_cluster} ({ref_label}) … ", end="", flush=True)
        try:
            res_df, fit_res, class_names = fit_model(
                df_posts, ref_cluster, use_post=use_post, use_ha=use_ha
            )
            save_results(res_df, fit_res, class_names, ref_cluster, out_folder)

            n_sig = res_df["Significant"].sum()
            print(f"done  (sig={n_sig}, R2={fit_res.prsquared:.3f})")

            # collect significant rows for the aggregate file
            sig_rows = res_df[res_df["Significant"]].copy()
            sig_rows["Model"] = model_name
            all_significant[model_name].append(sig_rows)

        except Exception as e:
            print(f"ERROR: {e}")

    # Save one aggregate CSV of all significant results for this model type
    if all_significant[model_name]:
        agg = pd.concat(all_significant[model_name], ignore_index=True)
        agg_path = os.path.join(model_dir, f"{model_name}_all_significant.csv")
        agg.to_csv(agg_path, index=False)
        print(f"\n  Aggregate significant results saved -> {agg_path}")

print("\n\nAll done!")
