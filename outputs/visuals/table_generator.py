import matplotlib.pyplot as plt

# -----------------------------
# Data (fill in MAE if needed)
# -----------------------------
data = [
    ["Top 8", 27512.84, 16963.63],
    ["Top 5", 28122.43, 17486.46],
    ["Full", 28872.99, 17756.22],
    ["2x Sparse", 38852.68, 24645.71],
    ["5x Sparse", 56244.91, 36834.91],
]

columns = ["Variant", "RMSE", "MAE"]

# -----------------------------
# Create table
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center",
    cellLoc="center"
)

# -----------------------------
# Styling
# -----------------------------
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)

# Bold header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')

plt.title("Model Performance Comparison", fontsize=14, pad=10)

plt.tight_layout()
plt.savefig("performance_table.png", dpi=300)
plt.close()

print("Saved as performance_table.png")
