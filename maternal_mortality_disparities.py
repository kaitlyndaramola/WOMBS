import arviz as az
import numpy as np
import pymc as pm
from pymc.math import dot, invlogit, logit
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

if __name__ == "__main__":
    #df for pregnancy model
    df = pd.read_csv("C:/Users/kaitl/OneDrive/Desktop/IYSE/Project/Pregnancy-Associated_Mortality_20250413.csv")
    df['Related'] = df['Related'].str.lower().str.strip()
    
    df_preg = df[(df['Race/ethnicity'] != "All")&(df['Race/ethnicity'] != "Race/ethnicity")&(df['Related'] == "pregnancy-related")].copy()
    df_preg_race_encoded = pd.get_dummies(df_preg, columns = ['Race/ethnicity'], drop_first = False)
    print(len(df_preg)) #number of filtered data samples
    #print(df_preg.head(20))
    #print(df_race_encoded)
    #print(df_race['Race/ethnicity'].unique())
    #print(df_preg['Related'].unique())

    race_columns = []
    for col in df_preg_race_encoded.columns:
        if col.startswith("Race/ethnicity_"):
            race_columns.append(col)
    x = df_preg_race_encoded[race_columns].to_numpy()
    y = df_preg_race_encoded['Deaths'].to_numpy()

    '''
    #check if things are matching 
    for i, row in enumerate(x):
        print(row, y[i])
    '''  

    with pm.Model() as m:
        #priors
        sigma = pm.HalfNormal("sigma", sigma = 10)
        beta_black = pm.Normal("beta_black", mu = 0, sigma = 10) #1st i
        beta_white = pm.Normal("beta_white", mu = 0, sigma = 10) #4th i
        beta_asian = pm.Normal("beta_asian", mu = 0, sigma = 10) #0th
        beta_latina = pm.Normal("beta_latina", mu = 0, sigma = 10) #2nd
        beta_other = pm.Normal("beta_other", mu = 0, sigma = 10) #3rd
        #mu = pm.Normal("mu", alpha = 1, beta = 1)

        #comparisons
        pm.Deterministic("Black vs. White", beta_black - beta_white)
        pm.Deterministic("Black vs. Asian/Pacific Islander", beta_black - beta_asian)
        pm.Deterministic("Black vs. Latina", beta_black - beta_latina)
        pm.Deterministic("Black vs. Other", beta_black - beta_other)

        #equation
        mu = beta_black*x[:,1] + beta_white*x[:,4] + beta_asian*x[:,0]+ beta_latina*x[:,2] + beta_other*x[:,3]

        #likelihood
        likelihod = pm.Normal('Deaths', mu= mu, sigma = sigma, observed = y)

        trace = pm.sample(5000, tune=1000, chains = 4, cores=4)

        summary = az.summary(trace, 
                             var_names = ["sigma", "beta_black", 
                                          "beta_white", "beta_asian", 
                                          "beta_latina", "beta_other",
                                          "Black vs. White", 
                                          "Black vs. Asian/Pacific Islander", 
                                          "Black vs. Latina", "Black vs. Other"], 
                                          hdi_prob=0.95)
        print(summary)
        summary.to_csv("project_posterior.csv")

        az.plot_posterior(
            trace,
            var_names = ["sigma", "beta_black", 
                         "beta_white", "beta_asian", 
                         "beta_latina", "beta_other"],
                         hdi_prob = 0.95,
                         figsize = (10,6),
                         kind="kde"
        )
        fig = plt.gcf()
        fig.suptitle("Posterior Distributions by Race", fontsize = 14)
        plt.tight_layout()
        plt.show()
        #plt.savefig("posterior_by_race_plot.png", dpi=300, bbox_inches = "tight")

#comparison plots to see stat sig of disaprities between black women and other races
        az.plot_posterior(trace,
                          var_names = ["Black vs. White", 
                                       "Black vs. Asian/Pacific Islander", 
                                       "Black vs. Latina", "Black vs. Other"],
                        hdi_prob = 0.95,
                        figsize = (10,6),
                        kind="kde")
        fig = plt.gcf()
        fig.suptitle("Mean Difference Posterior Distributions Between Black Women & Other Racial Groups", fontsize = 14)
        plt.tight_layout()
        plt.show()
        #plt.savefig("posterior_diff_plots.png", dpi=300, bbox_inches = "tight")


#save for Linkedin portfolia - Time series forecasting predict death rates of black women.
#https://areding.github.io/6420-pymc/unit10/Unit10-sunspots.html
#filtered well, but multiple diverenges look there, fixed
        df_filtered = df[df['Related'] == 'pregnancy-related']
        df_filtred = df_filtered[(df_filtered['Race/ethnicity'] != "All")&(df['Race/ethnicity'] != "Race/ethnicity")]
        df_grouped = df_filtered.groupby(["Year", "Race/ethnicity"])['Deaths'].sum().reset_index()
        df_pivot = df_grouped.pivot(index = "Year", columns = "Race/ethnicity", values = "Deaths")
        df_pivot = df_pivot.drop(columns = ["Other", "Asian/Pacific Islander"], errors = "ignore")
        df_pivot = df_pivot.astype(int)
        print(df_pivot)

        y = df_pivot["Black non-Latina"].values
        year = df_pivot.index.values

        y_mean = y.mean()
        y = y - y_mean
        #time series model
        y_prev = y[:-1]
        y_obs = y[1:]


        with pm.Model() as time_series_model:
            intercept = pm.Normal("intercept", mu=0, sigma = 5)
            slope = pm.Normal("slope", mu=0, sigma = 0.5)
            #rho = pm.Normal("rho", mu=0, sigma = 5, shape=2)
            sigma_ts = pm.HalfNormal("sigma_ts", sigma = 5)
            #pm.AR("likelihood", rho=rho, sigma=sigma_ts, constant = True, observed = y)
            mu = intercept + slope * y_prev
            likelihood = pm.Normal("observed", mu=mu, sigma = sigma_ts, observed = y_obs)
            time_series_trace = pm.sample(5000, tune=1000, chains = 4, cores=4, target_accept = 0.95)
            time_series_summary = az.summary(
                time_series_trace, 
                var_names= ['intercept', 'slope', "sigma_ts"],
                hdi_prob = 0.95)
        print(time_series_summary)

        #posterior means
        intercept_mean = time_series_trace.posterior['intercept'].mean(("chain", "draw")).values.item()
        slope_mean = time_series_trace.posterior['slope'].mean(("chain", "draw")).values.item()

        #
        n_forecast = 10
        y_forecast = np.zeros(n_forecast +1)
        y_forecast[0] = y_obs[-1]

        for i in range(1, n_forecast+1):
            y_forecast[i] = intercept_mean + slope_mean * y_forecast[i-1]

        y_forecast = y_forecast[1:]

        y_forecast_real = y_forecast + y_mean

        prev_year = year[-1]
        forecast_years = np.arange(prev_year+1, prev_year+1+n_forecast)

        plt.figure(figsize=(10,6))
        plt.plot(year, y+y_mean, label = "Observed Deaths", marker = 'o')
        plt.plot(forecast_years, 
                 y_forecast_real, 
                 label = "Forecasted Deaths", 
                 marker = 'x', linestyle = "--", color = 'red')
        plt.title("Forecasted Pregnancy Related Deaths for Black Women in NYC")
        plt.xlabel("Year")
        plt.ylabel("Deaths")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()