# WOMBS
Bayesian analysis of NYC maternal mortality data (2016–2021) by race. Used PyMC to model disparities with Bayesian linear regression and forecast future death rates using time series modeling. Results highlight significant inequities and the value of probabilistic modeling.

#Overview
This project explores maternal mortality trends across racial and ethnic groups in New York City from 2016 to 2021 using publicly available data from the NYC Department of Mental Health & Hygiene. This analysis applies:
- Bayesian Linear Regression to quantify the disparities in mortality rates between each racial/ethnic group.
- Bayesian Time Series Forecasting to project future maternal deaths among Black women.
All posterior analyses, model siagnostics and visualizations were performed using PyMC, ArviZ and matplotlib.

Key Findings:
- Black women had the highest average maternal mortality rates in the dataset
- All racial comparisons with the exception of the "Other" category showed statistically significant disparities.
- Forecasting showed persistently elevated mortality rates for Black women with current practices.

Tools:
- Python 3.12.5
- PyMC
- ArviZ
- pandas
- matlplotlib
