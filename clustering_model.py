import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_clusters(df, k=4, exclude_ethnicity="Latvian"):
    df2 = df[df["Ethnicity"].str.lower() != exclude_ethnicity.lower()].copy()
    for col in ["Citizen of Latvia","Non-citizen of Latvia","Latvian refugee","Temporary protection of Latvia","Alternative status of Latvia","Other","Latvian stateless person"]:
        df2[f"Share_{col.replace(' ','_')}"] = df2[col] / df2["Together"]

    share_cols = [c for c in df2.columns if c.startswith("Share_")]
    agg = df2.groupby("Ethnicity")[share_cols].mean()
    X = StandardScaler().fit_transform(agg)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    agg["Cluster"] = labels
    return agg.reset_index()
