
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def forecast(df, ethnicity, future_years=[2025,2030]):
    sub = df[df["Ethnicity"]==ethnicity].sort_values("Year")
    X, y = sub[["Year"]].values, sub["Together"].values
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X,y)
    preds = model.predict(np.array(future_years).reshape(-1,1))
    return pd.DataFrame({
        "Ethnicity": ethnicity,
        "Year": future_years,
        "Forecast": preds.round().astype(int)
    })
