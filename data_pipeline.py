

import os
import uuid
import time
import pandas as pd
import requests


AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "<YOUR_KEY_HERE>")
AZURE_ENDPOINT        = os.getenv("AZURE_ENDPOINT", "https://your-resource.cognitiveservices.azure.com/")
AZURE_REGION          = os.getenv("AZURE_REGION", "canadaeast")

def translate_text_azure(text_list, from_lang="lv", to_lang="en"):
    """
    Uses Azure Translator to translate a list of texts.
    """
    if not text_list:
        return []

    url = f"{AZURE_ENDPOINT}/translate?api-version=3.0&from={from_lang}&to={to_lang}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_REGION,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }

    translations = []
    for txt in text_list:
        body = [{"text": str(txt)}]
        try:
            resp = requests.post(url, headers=headers, json=body)
            resp.raise_for_status()
            translated = resp.json()[0]["translations"][0]["text"]
            translations.append(translated)
        except Exception:
    
            translations.append(txt)
        time.sleep(0.05)  # throttle a bit
    return translations

def load_and_merge(folder_path: str) -> pd.DataFrame:
    dfs = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".csv", ".xlsx")):
            continue
        path = os.path.join(folder_path, fname)

        if fname.lower().endswith(".csv"):
            df = pd.read_csv(path, dtype=str)
        else:
            df = pd.read_excel(path, dtype=str)

        orig_cols = df.columns.tolist()
        new_cols  = translate_text_azure(orig_cols)
        df.columns = new_cols

        for col in df.select_dtypes(include=["object"]).columns:
            unique_vals = df[col].dropna().unique().tolist()
            if unique_vals:
                trans = translate_text_azure(unique_vals)
                df[col] = df[col].map(dict(zip(unique_vals, trans)))

        year = ''.join(filter(str.isdigit, fname))[:4]
        df["Year"] = int(year)

        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c].str.replace(" ",""), errors="ignore")
            except:
                pass

        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

 
    full["Together"] = full["Together"].astype(float)

    full["Total_Population_Year"] = full.groupby("Year")["Together"].transform("sum")

    full["Percent_of_Year"] = (full["Together"] / full["Total_Population_Year"]) * 100

    return full
