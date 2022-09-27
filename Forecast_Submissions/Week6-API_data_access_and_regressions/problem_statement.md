# Forecasting assignment: API Data access and Regression analysis

In this forecasting assignment you'll be using APIs to download
streamflow data from the USGS database as well as analyzing it
with a basic regression model using the `scikit-learn` package.

This assignment is worth 10 points, one for each of the code steps
listed in the template which have code that you need to modify,
along with a point for successfully submitting a forecast.


## High level checklist:
To successfully score points for this assignment you must:

 1. Submit a modified python script titled `forecast_Oct4_2022.py` to your homework repository. Note: the template script is already there, you just need to fill in the places where you are instructed to do so.
 2. This script must read the past 30 years of data for the Verde river stream gauge.k
 3. You must fit a regression model that takes the historical streamflow data and predicts the next weeks streamflow.
 4. Additionally, you will fit 2 linear regression models which are specifically taylored to taking data from the week of Sept 26 and producing forecasts for the week of Oct 3 and Oct 10, respectively.
 5. Submit your streamflow forecasts for weeks beginning 10/3/2022 10/10/2022 to the forecasting repo.
 
## Hints:
 - You can access the week of year as an integer with `dataframe.index.weekofyear` or `dataframe.index.isocalendar().week`, with the latter preferred.
 - You can select out a particular week of year with:
    ```
    woy = 33
    woy_filter = dataframe.index.isocalendar().week == woy
    df_woy = dataframe[woy_filter]
    ```
