import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
baseline_rmse = 28872.99
results = {
    "Full": 28872.99,
    "Top 8": 27512.84,
    "Top 5": 28122.43,
    "2x Sparse": 38852.68,
    "5x Sparse": 56244.91,
}

# -----------------------------
# Helper functions
# -----------------------------
def add_value_labels(ax, values, fmt="{:,.0f}", rotation=0, fontsize=10):
    for i, v in enumerate(values):
        ax.text(
            i,
            v,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )

def add_pct_labels(ax, values, fmt="{:+.1f}%"):
    for i, v in enumerate(values):
        va = "bottom" if v >= 0 else "top"
        offset = 0.4 if v >= 0 else -0.4
        ax.text(
            i,
            v + offset,
            fmt.format(v),
            ha="center",
            va=va,
            fontsize=10,
        )

# -----------------------------
# 1. Feature reduction chart
# -----------------------------
feature_labels = ["Full", "Top 8", "Top 5"]
feature_values = [results[label] for label in feature_labels]

plt.figure(figsize=(8, 5))
ax = plt.gca()
bars = ax.bar(feature_labels, feature_values)

ax.set_title("RMSE Under Feature Reduction", fontsize=14)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_xlabel("Model Variant", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

add_value_labels(ax, feature_values)

plt.tight_layout()
plt.savefig("rmse_feature_reduction.png", dpi=300)
plt.close()

# -----------------------------
# 2. Temporal sparsity chart
# -----------------------------
sparsity_labels = ["Full", "2x Sparse", "5x Sparse"]
sparsity_values = [results[label] for label in sparsity_labels]

plt.figure(figsize=(8, 5))
ax = plt.gca()
bars = ax.bar(sparsity_labels, sparsity_values)

ax.set_title("RMSE Under Temporal Sparsity", fontsize=14)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_xlabel("Sampling Condition", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

add_value_labels(ax, sparsity_values)

plt.tight_layout()
plt.savefig("rmse_temporal_sparsity.png", dpi=300)
plt.close()

# -----------------------------
# 3. Combined comparison chart
# -----------------------------
combined_labels = ["Top 8", "Top 5", "Full", "2x Sparse", "5x Sparse"]
combined_values = [results[label] for label in combined_labels]

plt.figure(figsize=(10, 5))
ax = plt.gca()
bars = ax.bar(combined_labels, combined_values)

ax.set_title("Impact of Feature Reduction vs Temporal Sparsity", fontsize=14)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_xlabel("Model Variant", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

add_value_labels(ax, combined_values)

plt.tight_layout()
plt.savefig("rmse_combined_comparison.png", dpi=300)
plt.close()

# -----------------------------
# 4. Percent change from baseline
# -----------------------------
pct_labels = ["Top 8", "Top 5", "Full", "2x Sparse", "5x Sparse"]
pct_values = [((results[label] - baseline_rmse) / baseline_rmse) * 100 for label in pct_labels]

plt.figure(figsize=(10, 5))
ax = plt.gca()
bars = ax.bar(pct_labels, pct_values)

ax.axhline(0, linewidth=1)
ax.set_title("Percent Change in RMSE Relative to Baseline", fontsize=14)
ax.set_ylabel("Percent Change (%)", fontsize=12)
ax.set_xlabel("Model Variant", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

add_pct_labels(ax, pct_values)

plt.tight_layout()
plt.savefig("rmse_percent_change.png", dpi=300)
plt.close()

print("Charts saved:")
print("- rmse_feature_reduction.png")
print("- rmse_temporal_sparsity.png")
print("- rmse_combined_comparison.png")
print("- rmse_percent_change.png")
