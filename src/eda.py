import floret
import umap
import numpy as np
import pandas as pd

model = floret.load_model("models/hufloret_.bin")

df_names = pd.read_csv("data/raw/persons.csv", sep=",")
df_names = df_names.astype(str)
df_orgs = pd.read_csv("data/raw/institutions.csv", sep=",")
df_orgs = df_orgs.astype(str)

id2name = dict(zip(df_names["person_id"], df_names["name"]))
id2org = dict(zip(df_orgs["institution_id"], df_orgs["name"]))

print(len(id2name.keys()), len(id2org.keys()))


##################################################################################################
#####                              Helper functions                                          #####
##################################################################################################
def vectorize_name(name):
    name = name.split()
    name = [n.strip().lower() for n in name]
    vecs = [model.get_word_vector(w) for w in name]
    return np.mean(vecs, axis=0)


def cosine_similarity(a, b):
    """Returns cosine similarity btw vectors a and b"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(a))


def cosine_similarity_matrix(A):
    """Returns the cosine similarity matrix of matrix A"""
    n = np.dot(A, A.T)
    p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
    p2 = np.sqrt(np.sum(A**2, axis=1))[np.newaxis, :]
    return n / (p1 * p2)


##################################################################################################
#####                                Similarity matrices                                     #####
##################################################################################################
name_matrix = [vectorize_name(n) for n in id2name.values()]
org_matrix = [vectorize_name(n) for n in id2org]

name_sim = cosine_similarity_matrix(np.asarray(name_matrix))
with open("data/interim/name_similarity.npy", "wb") as outfile:
    np.save(outfile, name_sim)
org_sim = cosine_similarity_matrix(np.asarray(org_matrix))
with open("data/interim/org_similarity.npy", "wb") as outfile:
    np.save(outfile, org_sim)
##################################################################################################
#####                                       Embed strings                                    #####
##################################################################################################
name_reducer = umap.UMAP(n_neighbors=5, min_dist=0.0, metric="cosine")
name_embeddings = name_reducer.fit(name_matrix)

org_reducer = umap.UMAP(n_neighbors=5, min_dist=0.0, metric="cosine")
org_embeddings = org_reducer.fit(org_matrix)

name_df = {"nid": [], "name": [], "x": [], "y": []}
for i in range(len(name_embeddings.embedding_)):
    name_df["nid"].append(list(id2name.keys())[i])
    name_df["name"].append(list(id2name.values())[i])
    name_df["x"].append(name_embeddings.embedding_[i][0])
    name_df["y"].append(name_embeddings.embedding_[i][1])

name_df = pd.DataFrame(name_df)
name_df.to_csv("data/interim/name_umap.csv", sep=",")


org_df = {"nid": [], "name": [], "x": [], "y": []}
for i in range(len(org_embeddings.embedding_)):
    org_df["nid"].append(list(id2org.keys())[i])
    org_df["name"].append(list(id2org.values())[i])
    org_df["x"].append(org_embeddings.embedding_[i][0])
    org_df["y"].append(org_embeddings.embedding_[i][1])

org_df = pd.DataFrame(org_df)
org_df.to_csv("data/interim/org_umap.csv")
