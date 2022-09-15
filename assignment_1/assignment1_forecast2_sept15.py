#%%
import numpy as np

filename =('https://raw.githubusercontent.com/HAS-Tools-Fall2022'
'/Course-Materials22/main/data/verde_river_daily_flow_cfs.csv')
flows = np.loadtxt(
filename, # The location of the text file
delimiter=',', # character which splits data into groups
usecols=1 # Just take column 1, which is the flows
)
print(flows)
print() #linebreak
print('No of days of data:', len(flows)) #no. of days
#%%
flow_last2weeks = flows[-14:] #flow for the last 2 weeks of the month
flow_penultimateweek = flows[-14:-7] #flow for the penultimate week
flow_lastweek = flows[-7:] #flow for final week
print('Flow for the last two weeks flow rates were', flow_last2weeks)
print() #linebreak
# %%
avg_penultimateweek = np.average(flow_penultimateweek)
avg_lastweek = np.average(flow_lastweek)
change_twoweeks = avg_penultimateweek - avg_lastweek
print('Average change in flow between the last two weeks is', change_twoweeks)
#the decrease by 109.91 in flow is due to subsiding monsoon rains
# %%
#Since we are still getting erratic rains, I expect that the flow might increase in week 1 and decrease in week2
#I would suggest that flow increases by 109 in week1 (same change as between last 2 weeks)
week1forecast_sept13 = avg_lastweek + change_twoweeks
print('Forecast for Week 1 from Sept 13 in (ft3/s) is', round(week1forecast_sept13))
#by week 2 the monsoons would be expected to subside
#so I would suggest flow decrease by 1/4 of the increase rate in week1
week2forecast_sept20 = week1forecast_sept13 - (change_twoweeks/4)
print('Forecast for Week 2 from Sept 20 (ft3/s) is', round(week2forecast_sept20))
# %%
