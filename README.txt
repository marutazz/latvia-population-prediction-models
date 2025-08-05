Dependencies
•	Python 3.8+
•	pandas
•	numpy
•	scikit-learn
•	matplotlib
•	streamlit
•	tensorflow
•	requests
•	openpyxl
Project structure:
├── app.py                           
├── data_pipeline.py                 
├── clustering_model.py              
├── linear_regression_model.py       
├── random_forest_model.py           
├── neural_network_model.py          
├── latvia_population_translated_azure.csv

Data pipeline (data_pipeline.py): loads, translates (LV→EN) and merges multiple  datasets into a single DataFrame with feature engineering (total population per year, percentage share).
Clustering (clustering_model.py): computes K-Means clusters on non‑Latvian citizenship shares to identify groups like Integrated Minorities, Split Status Groups, Large Historical Minority, Protected Populations. The reason to not include Latvians was to not interrupt the cluster

Forecasting models:
o	Linear Regression (linear_regression_model.py)
o	Random Forest (random_forest_model.py)
o	Neural Network (neural_network_model.py)
Interactive dashboard (app.py): a Streamlit app that lets you explore clustering results and forecast population trends by ethnicity.
