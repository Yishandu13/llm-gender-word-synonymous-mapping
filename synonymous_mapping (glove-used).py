"""
Install dependencies and import requirements
"""

!pip install pandas numpy
!pip install gensim openpyxl

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
from google.colab import files

"""
Upload Excel file containing male/female words (The format set here is male/female words stored in a different sheet of a table). In addition, there are separate sheet (gender pairs) containing gender words in pair for model training.
"""

uploaded = files.upload()

"""
embedding model: GloVe
"""

model = api.load("glove-wiki-gigaword-100")

# Getting the uploaded Excel file
excel_file = list(uploaded.keys())[0]

# read male/female word list
male_df = pd.read_excel(excel_file, sheet_name='male_word')
female_df = pd.read_excel(excel_file, sheet_name='female_word')
try:
    pairs_df = pd.read_excel(excel_file, sheet_name='gender_pairs')  # read data for training
except:
    pairs_df = pd.DataFrame(columns=['male_word', 'female_word'])

# Cleaning the word list
male_words = male_df.iloc[:, 0].dropna().astype(str).tolist()
female_words = female_df.iloc[:, 0].dropna().astype(str).tolist()

print(f"📘 Count of male words: {len(male_words)}")
print(f"📕 Count of female words: {len(female_words)}")
print(f"🔁 Count of gender words in pair: {len(pairs_df)}")

# Prioritise the use of gender_pairs to build gender_direction
vec_diffs = []
for idx, row in pairs_df.iterrows():
    m, f = row['male_word'], row['female_word']
    if m in model and f in model:
        vec_diffs.append(model[f] - model[m])
    else:
        print(f"⚠️ Skip missing word vector pairing: {m} - {f}")

if vec_diffs:
    gender_direction = np.mean(vec_diffs, axis=0)
    print("Gender direction vectors constructed based on gender_pairs")
else:
    print("⚠️ Use fallback: all female-male average")
    male_vecs = [model[w] for w in male_words if w in model]
    female_vecs = [model[w] for w in female_words if w in model]
    gender_direction = np.mean(female_vecs, axis=0) - np.mean(male_vecs, axis=0)

# Gender mapping function
def get_gender_counterpart(word, direction, topn=10):
    if word not in model:
        return None
    new_vec = model[word] + direction
    similar = model.similar_by_vector(new_vec, topn=topn)
    for candidate, _ in similar:
        if candidate.lower() != word.lower():
            return candidate
    return None

# male → female
male_to_female = {}
for w in male_words:
    mapped = get_gender_counterpart(w, gender_direction)
    male_to_female[w] = mapped

# female → male
female_to_male = {}
for w in female_words:
    mapped = get_gender_counterpart(w, -gender_direction)
    female_to_male[w] = mapped

male_df_out = pd.DataFrame.from_dict(male_to_female, orient='index', columns=['female_synonym'])
male_df_out.index.name = 'male_word'
male_df_out.reset_index(inplace=True)

female_df_out = pd.DataFrame.from_dict(female_to_male, orient='index', columns=['male_synonym'])
female_df_out.index.name = 'female_word'
female_df_out.reset_index(inplace=True)

male_df_out.to_csv('male_to_female_mapping.csv', index=False)
female_df_out.to_csv('female_to_male_mapping.csv', index=False)

files.download('male_to_female_mapping.csv')
files.download('female_to_male_mapping.csv')
