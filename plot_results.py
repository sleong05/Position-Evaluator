# plot_results.py
import matplotlib.pyplot as plt
import numpy as np

models = [
    'Material\nBaseline',
    'CNN v1\n(12 planes\n3 epochs)',
    'CNN v2\n(6 planes\n3 epochs)',
    'CNN v2\nFine-tuned\n(10 epochs)',
    'CNN\nPositional\n(10 epochs)',
]

maes = [3.74, 2.507, 2.319, 2.177, 2.235]

colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c', '#5cb85c']

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(models, maes, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

# add value labels on top of each bar
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f'{mae:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Test MAE (pawns)', fontsize=13)
ax.set_title('Chess Position Evaluation — Test MAE by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 4.5)
ax.axhline(y=2.177, color='green', linestyle='--', alpha=0.4, label='Best model (2.177)')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results_chart.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved to results_chart.png")