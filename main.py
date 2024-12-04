#%%
# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# Import classes from other files
from network import Network
from common_methods import CommonMethods
from iterative_ip import NodalClearing, NodalIP
from bilevel_ip import BilevelIP
from nodal_ip import BilevelNodalIP


#%% Main script
# Initialize investment planning model
seed = 38
timelimit = 600
budget = 1000
carbontax = 60


#%% Main from nodal_ip.py
# Intended initialization of chosen_hours:
chosen_days = [190] # if it should be random then do: chosen_days = None
chosen_days = range(365)
chosen_hours = list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24))

## Initialization of small model for testing:
n_hours = 3
first_hour = 19
chosen_hours = list('T{0}'.format(i) for i in range(first_hour, first_hour+n_hours))

# Run nodal clearing to obtain upper bounds for lambda
nc_org = NodalClearing(timelimit=timelimit, carbontax=carbontax, seed=seed, chosen_hours=chosen_hours)
nc_org.build_model()
nc_org.run()
price_forecast = nc_org.data.lambda_ # Upper bounds for lambda in the investment planning model

ip = BilevelNodalIP(chosen_hours=chosen_hours, budget=budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd_ub=price_forecast)

# Build model
ip.build_model()

# Run optimization
ip.run()

# Display results
ip.display_results()
#ip.plot_network()
#ip.plot_supply_demand_curve('T1')
#ip.plot_prices()
# Save investment planning data in a dataframe and then in a csv file
investment_values_df = pd.DataFrame(ip.data.investment_values)
investment_values_df.to_csv('''results/investment_values,hours={n_hours},
                            budget={budget},
                            carbontax={carbontax},
                            runtime={runtime}.csv'''.format(n_hours=n_hours,
                                                            budget=budget,
                                                            carbontax=carbontax,
                                                            runtime=ip.model.Runtime,
                                                            )
                            )

#%% Main from bilevel_ip.py
# Carbon TAX price: https://www.statista.com/statistics/1322214/carbon-prices-european-union-emission-trading-scheme/
# Carbon TAX price: https://www.eex.com/en/market-data/emission-allowances/eua-auction-results

# Intended initialization of chosen_hours:
chosen_days = [190] # if it should be random then do: chosen_days = None
chosen_days = range(365)
chosen_hours = list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24))

## Initialization of small model for testing:
n_hours = 3
first_hour = 19
chosen_hours = list('T{0}'.format(i) for i in range(first_hour, first_hour+n_hours))

# Initialize investment planning model
ip = BilevelIP(chosen_hours=chosen_hours, budget=budget, timelimit=timelimit, carbontax=carbontax, seed=seed)

# Build model
ip.build_model()
# Run optimization
ip.run()
# Display results
ip.display_results()

ip.plot_supply_demand_curve('T1')

#%% Main from iterative_ip.py
# Model parameters
hours = 365*24
expected_NPV = []
actual_NPV = []

# Create nodal clearing instance without new investments for a price forecast
nc_org = NodalClearing(hours=hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
nc_org.build_model()
nc_org.run()
nc_org.plot_prices()
price_forecast = nc_org.data.lambda_
p_forecast = pd.DataFrame(price_forecast)

# %%
budgets = np.linspace(0, 2000, 21)
for budget in budgets:
    ip = NodalIP(hours=hours, budget = budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd=price_forecast, invest_bound=100)
    ip.build_model()
    ip.run()
    ip.display_results()
    investments=ip.data.investment_values
    expected_NPV.append(ip.data.objective_value)

    # Create nodal clearing instance with new investments
    nc = NodalClearing(hours=hours, timelimit=timelimit, carbontax=carbontax, seed=seed, P_investment=investments)
    nc.build_model()
    nc.run()
    nc.display_results()
    actual_NPV.append(nc.data.npv)

# %%
plt.plot(budgets, expected_NPV, marker = 'o', label='Expected NPV')
plt.plot(budgets, actual_NPV, marker = 'd', label='Actual NPV')
# plt.xscale('log')
plt.xlabel('Budget [M€]')
# plt.yscale('log')
plt.ylabel('NPV [M€]')
plt.legend()
plt.show()