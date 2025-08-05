
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from data_pipeline import load_and_merge
from clustering_model import compute_clusters
from linear_regression_model import forecast as lr_forecast
from random_forest_model import forecast as rf_forecast
from neural_network_model import forecast as nn_forecast

st.set_page_config(layout="wide")
st.title("🇱🇻 Latvia Population ML Dashboard")


df = pd.read_csv("latvia_population_translated_azure.csv")

mode = st.sidebar.selectbox("Choose analysis", 
    ["Clustering","Linear Regression","Random Forest","Neural Network"])

if mode == "Clustering":
    st.header("K-Means Clustering of Non-Latvian Share Profiles")
    k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
    
    clusters = compute_clusters(df, k=k)
    
    cluster_name_map = {
        0: "Integrated Minorities",
        1: "Split Status Groups",
        2: "Large Historical Minority",
        3: "Protected Populations"
    }
    clusters["Cluster_Category"] = clusters["Cluster"].map(cluster_name_map)
    

    share_cols = [c for c in clusters.columns if c.startswith("Share_")]
    
    for cid, group in clusters.groupby("Cluster"):
        cat_name = cluster_name_map[cid]
        st.subheader(f"{cat_name} (Cluster {cid})")
        
        profile = group[share_cols].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        profile.plot.bar(ax=ax)
        
        ax.set_title(f"{cat_name}")
        ax.set_ylabel("Average Share")
        ax.set_xlabel("Status Category")
        
    
        xticks = [name.replace("Share_","") for name in share_cols]
        ax.set_xticklabels(
            xticks,
            rotation=30,      
            ha='right',      
            fontsize=9       
        )
        
        # percent y‐axis
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        st.pyplot(fig)

else:

    st.header(f"{mode} Forecast per Ethnicity")
    
 
    all_eth = sorted(df["Ethnicity"].unique().tolist())
    eth = st.sidebar.selectbox("Select ethnicity", all_eth)
    

    hist = df[df["Ethnicity"]==eth].sort_values("Year")
    X_hist = hist[["Year"]].values
    y_hist = hist["Together"].values
    

    if mode == "Linear Regression":
        preds = lr_forecast(df, eth)  
   
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X_hist, y_hist)
        y_fit = m.predict(X_hist)
        y_future = m.predict(preds[["Year"]].values.reshape(-1,1))
        
    elif mode == "Random Forest":
        preds = rf_forecast(df, eth)
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_hist, y_hist)
        y_fit = m.predict(X_hist)
        y_future = m.predict(preds[["Year"]].values.reshape(-1,1))
        
    else: 
        preds = nn_forecast(df, eth)
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        sc_y = StandardScaler().fit(y_hist.reshape(-1,1))
        sc_x = StandardScaler().fit(X_hist)
        Xs, ys = sc_x.transform(X_hist), sc_y.transform(y_hist.reshape(-1,1))
        m = Sequential([
            Dense(16, activation="relu", input_shape=(1,)),
            Dense(8, activation="relu"),
            Dense(1)
        ])
        m.compile("adam","mse")
        m.fit(Xs, ys, epochs=100, verbose=0)
        y_fit = sc_y.inverse_transform(m.predict(Xs)).flatten()
        y_future = preds["Forecast"].values 
    
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(hist["Year"], hist["Together"], color="navy", label="Actual")
    ax.plot(hist["Year"], y_fit, "r--", label="Fitted")
    ax.plot(preds["Year"], y_future, "ro--", label="Forecast")
    ax.set_xlabel("Year"); ax.set_ylabel("Total Population")
    ax.set_title(f"{eth} — {mode}")
    ax.grid(True); ax.legend()
    st.pyplot(fig)
    
    out = preds.copy()
    out["Model"] = mode
    st.subheader("Forecast values")
    st.dataframe(out, use_container_width=True)
