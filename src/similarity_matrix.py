import numpy as np
import pandas as pd

###############################################################################################
#####                                     Persons                                         #####
###############################################################################################
name_matrix = np.load("data/interim/name_similarity.npy")
df_names = pd.read_csv("data/raw/persons.csv", sep=",")
df_names = df_names.astype(str)
id2name = dict(zip(df_names["person_id"], df_names["name"]))
wds = list(id2name.values())
threshold_name = (name_matrix < 1.0) * name_matrix
threshold_name = np.argwhere(threshold_name > 0.94)

similar_names = {"name1": [],
                 "name2": [],
                 "similarity": []}
for e in threshold_name:
    wd1 = wds[e[0]]
    wd2 = wds[e[1]]
    if wd1 != wd2:
        sim = name_matrix[e[0]][e[1]]
        similar_names["name1"].append(wd1)
        similar_names["name2"].append(wd2)
        similar_names["similarity"].append(sim)
name_df = pd.DataFrame(similar_names)
name_df.to_csv("data/processed/name_candidates.csv", sep=",")
###############################################################################################
#####                                  Organizations                                      #####
###############################################################################################
org_matrix = np.load("data/interim/org_similarity.npy")
df_orgs = pd.read_csv("data/raw/institutions.csv", sep=",")
df_orgs = df_orgs.astype(str)
id2org = dict(zip(df_orgs["institution_id"], df_orgs["name"]))
wds = list(id2org.values())

threshold_org = (org_matrix < 1.0) * org_matrix
threshold_org = np.argwhere(threshold_org > 0.92)

similar_orgs = {"name1": [],
                 "name2": [],
                 "similarity": []}
for e in threshold_org:
    wd1 = wds[e[0]]
    wd2 = wds[e[1]]
    if wd1 != wd2:
        sim = org_matrix[e[0]][e[1]]
        similar_orgs["name1"].append(wd1)
        similar_orgs["name2"].append(wd2)
        similar_orgs["similarity"].append(sim)

org_df = pd.DataFrame(similar_orgs)
org_df.to_csv("data/processed/org_candidates.csv", sep=",")
