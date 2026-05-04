import matplotlib.pyplot as plt
import numpy as np

# ── Figure 1: Recall@K Small vs Large (already have this) ──
metrics = ['Recall@5', 'Recall@10', 'Recall@20']
small = [0.1219, 0.2315, 0.3743]
large = [0.0692, 0.1021, 0.1511]

x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - 0.2, small, width=0.4, label='299 games', color='steelblue')
ax.bar(x + 0.2, large, width=0.4, label='3300+ games', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Recall Score')
ax.set_title('Recall@K: Small vs Large Dataset')
ax.legend()
ax.set_ylim(0, 0.45)
for i, (s, l) in enumerate(zip(small, large)):
    ax.text(i - 0.2, s + 0.005, f'{s}', ha='center', fontsize=8)
    ax.text(i + 0.2, l + 0.005, f'{l}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('figure1_recall_comparison.png', dpi=150)
plt.show()

# ── Figure 2: Cosine vs Neural Autoencoder (already have this) ──
games = ['Guilty Gear', 'Elden Ring', 'Hollow Knight', 'Overwatch', 'RE4']
cosine = [0.853, 0.879, 0.972, 0.878, 0.798]
neural = [0.978, 0.999, 0.999, 0.990, 0.991]

x = np.arange(len(games))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, cosine, width=0.4, label='Cosine Similarity', color='steelblue')
ax.bar(x + 0.2, neural, width=0.4, label='Neural Autoencoder', color='teal')
ax.set_xticks(x)
ax.set_xticklabels(games, rotation=15)
ax.set_ylabel('Top-1 Similarity Score')
ax.set_title('Cosine vs Neural Autoencoder: Top-1 Similarity Scores')
ax.legend()
ax.set_ylim(0, 1.1)
for i, (c, n) in enumerate(zip(cosine, neural)):
    ax.text(i - 0.2, c + 0.01, f'{c}', ha='center', fontsize=8)
    ax.text(i + 0.2, n + 0.01, f'{n}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('figure2_method_comparison.png', dpi=150)
plt.show()

# ── Figure 3: Recall@K line chart showing improvement as K increases ──
k_values = [5, 10, 20]
small_recall = [0.1219, 0.2315, 0.3743]
large_recall = [0.0692, 0.1021, 0.1511]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, small_recall, marker='o', linewidth=2, 
        label='299 games', color='steelblue')
ax.plot(k_values, large_recall, marker='s', linewidth=2, 
        label='3300+ games', color='coral')

for k, s, l in zip(k_values, small_recall, large_recall):
    ax.annotate(f'{s}', (k, s), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=9, color='steelblue')
    ax.annotate(f'{l}', (k, l), textcoords="offset points", 
                xytext=(0, -15), ha='center', fontsize=9, color='coral')

ax.set_xlabel('K (number of recommendations)')
ax.set_ylabel('Recall Score')
ax.set_title('Recall@K Improvement as K Increases')
ax.set_xticks(k_values)
ax.set_xticklabels(['Top 5', 'Top 10', 'Top 20'])
ax.legend()
ax.set_ylim(0, 0.45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_recall_line.png', dpi=150)
plt.show()

print("All 3 figures saved!")