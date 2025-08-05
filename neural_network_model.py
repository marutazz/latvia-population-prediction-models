import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def forecast(df, ethnicity, future_years=[2025, 2030]):
    sub = df[df["Ethnicity"] == ethnicity].sort_values("Year")
    X, y = sub[["Year"]].values, sub["Together"].values.reshape(-1, 1)

 
    sc_year = StandardScaler().fit(X)
    sc_pop  = StandardScaler().fit(y)
    X_s, y_s = sc_year.transform(X), sc_pop.transform(y)


    m = Sequential([
        Dense(16, activation="relu", input_shape=(1,)),
        Dense(8, activation="relu"),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    m.fit(X_s, y_s, epochs=100, verbose=0)

    fX_s = sc_year.transform(np.array(future_years).reshape(-1, 1))
    p_s  = m.predict(fX_s)
    preds = sc_pop.inverse_transform(p_s).flatten().round().astype(int)

    return pd.DataFrame({
        "Ethnicity": ethnicity,
        "Year": future_years,
        "Forecast": preds
    })
