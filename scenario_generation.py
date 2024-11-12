import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.path.dirname(__file__)
xls = pd.ExcelFile(cwd + '/data/grid_data.xlsx')

## Loading data from Excel sheets
system_demand = pd.read_excel(xls, 'demand')

demand_flux = pd.read_excel(cwd + '/data/load2023.xlsx').iloc[1:, :]
demand_flux.columns = ['Datetime','Hour','Demand','Price']
demand_flux['Datetime'] = pd.to_datetime(demand_flux['Date'])
demand_flux['Hour'] = demand_flux['Hour'].astype(int)
demand_flux['Demand'] = demand_flux['Demand'].astype(float)
demand_flux['Price'] = demand_flux['Price'].astype(float)

# Production profiles for wind and solar
offshore_hourly_2019 = pd.DataFrame(pd.read_csv(cwd + '/data/offshore_hourly_2019.csv', skiprows=3)['electricity']) # Wind production profiles for a 1 MW wind farm [pu]
onshore_hourly_2019 = pd.DataFrame(pd.read_csv(cwd + '/data/onshore_hourly_2019.csv', skiprows=3)['electricity']) # Wind production profiles for a 1 MW wind farm [pu]
solar_hourly_2019 = pd.DataFrame(pd.read_csv(cwd + '/data/pv_hourly_2019.csv', skiprows=3)['electricity']) # Solar production profiles for a 1 MW solar farm [pu]
offshore_hourly_2019['Hour'] = demand_flux['Hour'].values
onshore_hourly_2019['Hour'] = demand_flux['Hour'].values
solar_hourly_2019['Hour'] = demand_flux['Hour'].values

fig, axs =  plt.subplots(2,2,figsize=(12,10))

for i in range(0, 365):
    off = offshore_hourly_2019.iloc[i*24:(i+1)*24]
    axs[0,0].plot(off['Hour'], off['electricity'], alpha=0.05, color='blue')
    on = onshore_hourly_2019.iloc[i*24:(i+1)*24]
    axs[0,1].plot(on['Hour'], on['electricity'], alpha=0.05, color='blue')
    sol = solar_hourly_2019.iloc[i*24:(i+1)*24]
    axs[1,0].plot(sol['Hour'], sol['electricity'], alpha=0.05, color='blue')



sns.lineplot(data=offshore_hourly_2019, x='Hour', y= 'electricity', ax=axs[0,0], errorbar='sd', label='Hourly mean and standard deviation')
sns.lineplot(data=onshore_hourly_2019, x='Hour', y= 'electricity', ax=axs[0,1], errorbar='sd', label='Hourly mean and standard deviation')
sns.lineplot(data=solar_hourly_2019, x='Hour', y= 'electricity', ax=axs[1,0], errorbar=None, label='Hourly mean')
axs[0,0].legend(loc='upper right')
axs[0,0].set_title('Offshore wind production profile')
axs[0,0].set_xlabel('Hour')
axs[0,0].set_ylabel('Production [pu]')
axs[0,0].set_xlim(1,24)
axs[0,0].set_ylim(0,1)
axs[0,1].legend(loc='upper right')
axs[0,1].set_title('Onshore wind production profile')
axs[0,1].set_xlabel('Hour')
axs[0,1].set_ylabel('Production [pu]')
axs[0,1].set_xlim(1,24)
axs[0,1].set_ylim(0,1)
axs[1,0].legend(loc='upper right')
axs[1,0].set_title('Solar production profile')
axs[1,0].set_xlabel('Hour')
axs[1,0].set_ylabel('Production [pu]')
axs[1,0].set_xlim(1,24)
axs[1,0].set_ylim(0,1)

for i in range(0, 365):
    if i == 1:
        label = '2023 Historical Demands'
    else:
        label = ''
    daily_demand = demand_flux.iloc[i*24:(i+1)*24]
    axs[1,1].plot(daily_demand['Hour'], daily_demand['Demand'], alpha=0.1, color='blue', label=label)
axs[1,1].plot(range(1,25),system_demand['System_demand'], color='red', label='System demand (IEEE Bus 24 System)')
axs[1,1].set_xlabel('Hour')
axs[1,1].set_ylabel('Demand [MW]')
axs[1,1].set_title('Daily demand profiles')
axs[1,1].set_xlim(1,24)
#axs[1,1].set_ylim(0,1000)
axs[1,1].legend(loc='upper right')

plt.tight_layout(pad=3.0)

plt.show()


