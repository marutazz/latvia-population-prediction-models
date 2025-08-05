
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def forecast(df, ethnicity, future_years=[2025,2030]):
    sub = df[df["Ethnicity"]==ethnicity].sort_values("Year")
    X = sub[["Year"]].values
    y = sub["Together"].values
    model = LinearRegression().fit(X,y)
    preds = model.predict(np.array(future_years).reshape(-1,1))
    return pd.DataFrame({
        "Ethnicity": ethnicity,
        "Year": future_years,
        "Forecast": preds.round().astype(int)
    })
