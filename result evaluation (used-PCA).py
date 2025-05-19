"""Evaluation with PCA"""

!pip install openpyxl --quiet

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from google.colab import files

uploaded = files.upload() # upload result of gender_synonym_map

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

excel_file = list(uploaded.keys())[0]
df = pd.read_excel(excel_file)

male_words = df['male_word'].dropna().astype(str).tolist()
female_words = df['female_word'].dropna().astype(str).tolist()

labels = []
male_vecs = []
female_vecs = []

for m, f in zip(male_words, female_words):
    if m in model and f in model:
        male_vecs.append(model[m])
        female_vecs.append(model[f])
        labels.append((m, f))
    else:
        print(f"⚠️ missing word vector: {m} - {f}")

# Perform PCA downscaling
all_vecs = np.array(male_vecs + female_vecs)
pca = PCA(n_components=2)
proj = pca.fit_transform(all_vecs)

# Showing the explained variance ratio
explained_var = pca.explained_variance_ratio_
print("📊 PCA Explained Variance Ratio (PC1, PC2):", explained_var)
print(f"Cumulative proportion of variance explained: {explained_var.sum():.2%}")

plt.figure(figsize=(12, 8))
n = len(labels)

gender_direction = np.mean(np.array(female_vecs) - np.array(male_vecs), axis=0)
delta_vectors = []
cosine_scores = []

for i in range(n):
    delta = female_vecs[i] - male_vecs[i]
    delta_vectors.append(delta)
    score = cosine_similarity([delta], [gender_direction])[0][0]
    cosine_scores.append(score)

# Statistical outliers (cos too different from average)
mean_cos = np.mean(cosine_scores)
std_cos = np.std(cosine_scores)
outlier_indices = [i for i, score in enumerate(cosine_scores) if abs(score - mean_cos) > 2 * std_cos]

print(f"📈 Average cosine similarity: {mean_cos:.4f}, standard deviation: {std_cos:.4f}")
print(f"🚨 Counts of outliers（|score - mean| > 2σ）: {len(outlier_indices)}")

# visualisation
for i in range(n):
    color = 'orange' if i in outlier_indices else 'gray'

    # male, female scatters
    plt.scatter(proj[i, 0], proj[i, 1], color='blue')
    plt.scatter(proj[i+n, 0], proj[i+n, 1], color='red')

    # arrow
    plt.arrow(proj[i, 0], proj[i, 1],
              proj[i+n, 0] - proj[i, 0],
              proj[i+n, 1] - proj[i, 1],
              color=color, alpha=0.5, head_width=0.08, length_includes_head=True)

    # outlier
    if i in outlier_indices:
        plt.text(proj[i, 0], proj[i, 1], labels[i][0], fontsize=9, color='black')
        plt.text(proj[i+n, 0], proj[i+n, 1], labels[i][1], fontsize=9, color='black')

# plt
plt.title('PCA Projection of Gender Word Pairs')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
