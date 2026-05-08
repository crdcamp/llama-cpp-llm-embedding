# data from https://www.bowdoin.edu/studentaffairs/academic-honesty/examples/mosaic/index.shtml
# %% Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# %% Data
txt_original = "Contrast the condition into which all these friendly Indians are suddenly plunged now, with their condition only two years previous: martial law now in force on all their reservations; themselves in danger of starvation, and constantly exposed to the influence of emissaries from their friends and relations, urging them to join in fighting this treacherous government that had kept faith with nobody--neither with friend nor with foe."
txt_plagiarized = "Only two years later, all these friendly Sioux were suddenly plunged into new conditions, including starvation, martial law on all their reservations, and constant urging by their friends and relations to join in warfare against the treacherous government that had kept faith with neither friend nor foe."
txt_control = "Only two years later, all the money he won from lottery was gone."

txts = [txt_original, txt_plagiarized, txt_control]

# %%
unigram_count = CountVectorizer(encoding='latin-1', binary=False)
unigram_count_stop_remove = CountVectorizer(encoding='latin-1', binary=False, stop_words='english')

vecs = unigram_count.fit_transform(txts)

print(vecs.shape)
print(vecs[0].shape)

# %%

cos_sim = cosine_similarity(vecs[0], vecs)
print(cos_sim)
sim_sorted_doc_idx = cos_sim.argsort()
print(sim_sorted_doc_idx.shape)

# print the most similar doc; it's actually the original doc itself
print(txts[sim_sorted_doc_idx[0][len(txts)-1]])
print()

# print the second most similar doc; it's the most likely plagiarized one
print(txts[sim_sorted_doc_idx[0][len(txts)-2]])

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
X = np.array([1,2])
Y = np.array([2,2])
Z = np.array([2,4])

# calculate cosine similarity between [X] and [Y,Z]
# sending input as arrays would allow for calculating both cosine_sim(X,Y) and cosine_sim (X,Y)
cos_sim = cosine_similarity([X], [Y,Z])
print(cos_sim)

# calculate the entire cosie similarity matrix among X, Y, and Z
cos_sim = cosine_similarity([X, Y, Z])
print(cos_sim)
print()

# %%
print(f"{X.shape}, {Y.shape}, {Z.shape}")
