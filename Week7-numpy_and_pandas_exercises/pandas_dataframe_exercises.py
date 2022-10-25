#%%
# This script contains exercises on 
# manipulating Series and DataFrames with pandas
import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


#%% 
# Same data manipulation to get USGS streamflow as
# a pandas dataframe as before
def create_usgs_url(site_no, begin_date, end_date):
    return (
        f'https://waterdata.usgs.gov/nwis/dv?'
        f'cb_00060=on&format=rdb&referred_module=sw&'
        f'site_no={site_no}&'
        f'begin_date={begin_date}&'
        f'end_date={end_date}'
    )

def open_usgs_data(site, begin_date, end_date):
    url = create_usgs_url((site), begin_date, end_date)
    response = urllib.request.urlopen(url)
    df = pd.read_table(
        response,
        comment='#',
        skipfooter=1,
        delim_whitespace=True,
        names=['agency', 'site', 'date', 'streamflow', 'quality_flag'],
        index_col=2,
        parse_dates=True
    ).iloc[2:]

    # Now convert the streamflow data to floats and
    # the index to datetimes. When processing raw data
    # it's common to have to do some extra postprocessing
    df['streamflow'] = df['streamflow'].astype(np.float64)
    df.index = pd.DatetimeIndex(df.index)
    return df

def open_daymet_data(lat, lon, begin_date, end_date):
    args = {'lat':  lat, 'lon': lon, 'format': 'csv',
            'start': begin_date, 'end': end_date}
    query = urllib.parse.urlencode(args)
    url = f"https://daymet.ornl.gov/single-pixel/api/data?{query}"
    response = urllib.request.urlopen(url)
    df = pd.read_csv(response, header=6)
    datestring = (df['year'].astype(str) + df['yday'].astype(str))
    dates = pd.to_datetime(datestring, format='%Y%j')
    df.index = pd.DatetimeIndex(dates)
    return df

site = '09506000'
begin_date = '1992-09-25'
end_date = '2022-09-25'
lat = 34.4483605
lon = -111.7898705

verde_df = open_daymet_data(lat, lon, begin_date, end_date)
usgs_df = open_usgs_data(site, begin_date, end_date)
verde_df = verde_df.reindex(verde_df.index)
verde_df['streamflow'] = usgs_df['streamflow']
verde_df.head()

# %%
# 1. How do you see a quick summary of what is in `verde_df`?
verde_df.head()
# %%
# 2. How do you get a listing of the columns in `verde_df`?
verde_df.columns
# %%
# 3. How do you select the streamflow column in `verde_df`?
verde_df.streamflow
#%%
# 4. How do you plot the streamflow in `verde_df`?
verde_df.plot(x = "yday", y ="streamflow", kind = "scatter") 
#%%
# 5. How do you get the last streamflow value from `verde_df`?
verde_df["streamflow"].iloc[-1]
#%%
# 6. What is the mean streamflow value for the 30 year period?
verde_df["streamflow"].mean
#%%
# 7. What is the maximum value for the 30 year period?
verde_df["streamflow"].max
#%% 
# 8. How do you find the maximum streamflow value for each year?
verde_df.groupby(['year'])["streamflow"].max()

#%%
# 9. How do you make a scatter plot of 
#    `dayl (s)` versus `tmax (deg c)`?
# INFO: `dayl` is the day length in seconds and
#       `tmax` is the daily maximum temperature
verde_df.plot(x = "dayl (s)", y ="tmax (deg c)", kind = "scatter") 
#%%
# 10. How do you calculate (and plot) the mean
#     daily minimum temperature for each day of year? 
#     And plot it?
# INFO: Daily minimum temperature is in the column `tmin (deg c)`

tmin_mean = verde_df.groupby(['yday'])["tmin (deg c)"].mean()
print(tmin_mean)
#
#plot
tmin_mean.plot()
plt.xlabel('Day of Year')
plt.ylabel('Daily Min Temp, deg C')
#%%
# 11. What is the average value of all columns for October 10 
#     across all years?
# INFO: October 10 is the 283rd day of year
df_Oct10 = verde_df.loc[verde_df['yday'] == 283].mean()
print(df_Oct10)
#verde_df.groupby(["yday"].mean()
#%%
# ----------------------------------------------------------------------------
# ========== NEW DATASET =====================================================
# ----------------------------------------------------------------------------
# Loading data - here I provide a dataset for you
# to work with for the first set of exercises
#
# The iris dataset is a classic and very easy 
# multi-class classification dataset. It describes
# measurments of sepal & petal width/length for three
# different species of iris
d = load_iris()
iris_df = pd.DataFrame(d['data'], columns=d['feature_names'])
iris_df.index = pd.Series(
    pd.Categorical.from_codes(d.target, d.target_names),
    name='species'
)
iris_df.head()

# %%
# 12. How do you view the "unique" species in the `iris_df` index?
iris_df.index.unique()
# %%
# 13. How do you "locate" only rows for the `versicolor` species?
versicolor_df = iris_df.loc['versicolor']
print(versicolor_df)
# %%
# 14. How do you group by the 3 different species and take 
#     the mean across the whole dataframe?
iris_df.groupby(['species']).mean()
# %%
# 15. How do you make a scatter plot of the `sepal length (cm)` 
#     versus the `petal length (cm)` for the `versicolor`` species?
#
versicolor_df.plot(x= 'sepal length (cm)',y = 'petal length (cm)', kind = 'scatter')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm')
plt.title('Versicolor Data')
# BONUS OPTION: Do the same plot for `setosa` and `virginica` all on 
#               the same figure. Color them 'tomato', 'darkcyan', 
#               and 'darkviolet', respectively. 
#               Worth 1 point extra credit

#%%
#creating indices
setosa_df = iris_df.loc['setosa']
virginica_df = iris_df.loc['virginica']

plt.scatter(x= versicolor_df['sepal length (cm)'], y = versicolor_df['petal length (cm)'],  color = 'tomato', label = 'versicolor')
plt.scatter(x= setosa_df['sepal length (cm)'], y = setosa_df['petal length (cm)'], color = 'darkcyan', label = 'setosa')
plt.scatter(x= virginica_df['sepal length (cm)'], y = virginica_df['petal length (cm)'], color = 'darkviolet', label = 'virginica')
plt.legend()
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm')
plt.title('All Data')
# %%