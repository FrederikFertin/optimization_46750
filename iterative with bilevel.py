#%% Preamble
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import classes from other files
from iterative_ip import NodalClearing, NodalIP
from nodal_ip import BilevelNodalIP
import seaborn as sns
from time import time

# Set style for plots
sns.set_style("whitegrid")

#%% Main script
# Initialize investment planning model
seed = 38
timelimit = 600
carbontax = 60
## Initialization of small model for testing:
n_hours = 3
first_hour = 10
chosen_hours = list('T{0}'.format(i) for i in range(first_hour, first_hour+n_hours))

budget = 1000

chosen_days_list = [range(365), range(180), range(1)]
chosen_hours_list = [list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24)) for chosen_days in chosen_days_list]

nc_runtimes = []
ip_runtimes = []

for chosen_hours in chosen_hours_list:
    t_start = time()
    nc_org = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
    nc_org.build_model()
    nc_org.run()
    price_ub = nc_org.data.lambda_
    nc_runtimes.append(time()-t_start)
    
    t_start = time()
    ip = NodalIP(chosen_hours=chosen_hours, budget = budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd=price_ub)
    ip.build_model()
    ip.run()
    ip_runtimes.append(time()-t_start)

# Create nodal clearing instance without new investments for a price forecast
nc_org = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
nc_org.build_model()
nc_org.run()
price_ub = nc_org.data.lambda_

# %%
budgets = np.linspace(0, 2000, 11)
investment_bounds = [np.inf, 500, 200, 100]
expected_NPV = {str(bound): {} for bound in investment_bounds}
actual_NPV = {str(bound): {} for bound in investment_bounds}
optimal_NPV = {str(bound): {} for bound in investment_bounds}
optimal_investments = {str(bound): {} for bound in investment_bounds}

for budget in budgets:
    for invest_bound in investment_bounds:
        ip = NodalIP(chosen_hours=chosen_hours, budget = budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd=price_ub, invest_bound=invest_bound)
        ip.build_model()
        ip.run()
        investments=ip.data.investment_values
        max_npv=ip.data.objective_value
        expected_NPV[str(invest_bound)][str(budget)] = max_npv

        # Create nodal clearing instance with new investments
        nc = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed, P_investment=investments)
        nc.build_model()
        nc.run()
        min_npv=nc.data.npv
        actual_NPV[str(invest_bound)][str(budget)] = min_npv

        ip_bilevel = BilevelNodalIP(chosen_hours=chosen_hours,
                                    budget=budget,
                                    timelimit=timelimit,
                                    carbontax=carbontax,
                                    seed=seed,
                                    lmd_ub=price_ub,
                                    bounded=True,
                                    #min_npv=min_npv,
                                    #max_npv=max_npv,
                                    investments_guess=investments,
                                    invest_bound=invest_bound,
                                    )
        ip_bilevel.build_model()
        ip_bilevel.run()
        optimal_investments[str(invest_bound)][str(budget)] = ip_bilevel.data.investment_values
        optimal_NPV[str(invest_bound)][str(budget)] = ip_bilevel.data.objective_value

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.tight_layout(pad=5.0)
for i, bound in enumerate(investment_bounds):
    ax = axs[i//2, i%2]
    ax.plot(budgets, expected_NPV[str(bound)].values(), marker = 'o', label='Expected NPV')
    ax.plot(budgets, actual_NPV[str(bound)].values(), marker = 'd', label='Actual NPV')
    ax.plot(budgets, optimal_NPV[str(bound)].values(), marker = 'x', label='Optimal NPV')
    ax.set_xlabel('Budget [M€]')
    ax.set_ylabel('NPV [M€]')
    ax.set_xlim([0, 2000])
    if bound == np.inf:
        bound = 'Unlimited'
    ax.set_title('Investment bound: {0} MW'.format(bound))
    ax.legend()
plt.show()

expected_NPV_all_data = {str(bound): {} for bound in investment_bounds}
actual_NPV_all_data = {str(bound): {} for bound in investment_bounds}
optimal_NPV_all_data = {str(bound): {} for bound in investment_bounds}
actual_optimal_NPV = {str(bound): {} for bound in investment_bounds}

chosen_days = range(365)
chosen_hours = list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24))

# Create nodal clearing instance without new investments for a price forecast
nc_org = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
nc_org.build_model()
nc_org.run()
price_ub = nc_org.data.lambda_

for budget in budgets:
    for invest_bound in investment_bounds:
        ip = NodalIP(chosen_hours=chosen_hours, budget = budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd=price_ub, invest_bound=invest_bound)
        ip.build_model()
        ip.run()
        investments=ip.data.investment_values
        max_npv=ip.data.objective_value
        expected_NPV_all_data[str(invest_bound)][str(budget)] = max_npv

        # Create nodal clearing instance with new investments
        nc = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed, P_investment=investments)
        nc.build_model()
        nc.run()
        min_npv=nc.data.npv
        actual_NPV_all_data[str(invest_bound)][str(budget)] = min_npv

        nc2 = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed, P_investment=optimal_investments[str(invest_bound)][str(budget)])
        nc2.build_model()
        nc2.run()
        actual_optimal_NPV[str(invest_bound)][str(budget)] = nc2.data.npv

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.tight_layout(pad=5.0)
for i, bound in enumerate(investment_bounds):
    ax = axs[i//2, i%2]
    ax.plot(budgets, expected_NPV[str(bound)].values(), marker = 'o', label='Expected NPV')
    ax.plot(budgets, actual_NPV[str(bound)].values(), marker = 'd', label='Actual NPV')
    ax.plot(budgets, optimal_NPV[str(bound)].values(), marker = 'x', label='"Optimal" NPV')
    ax.plot(budgets, expected_NPV_all_data[str(bound)].values(), marker = 'o', label='Expected NPV (all data)', linestyle='dashed')
    ax.plot(budgets, actual_NPV_all_data[str(bound)].values(), marker = 'd', label='Actual NPV (all data)', linestyle='dashed')
    ax.set_xlabel('Budget [M€]')
    ax.set_ylabel('NPV [M€]')
    ax.set_xlim([0, 2000])
    if bound == np.inf:
        bound = 'Unlimited'
    ax.set_title('Investment bound: {0} MW'.format(bound))
    ax.legend()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10),)
fig.tight_layout(pad=5.0, rect=[0, 0, 1, 1])
for i, bound in enumerate(investment_bounds):
    ax = axs[i//2, i%2]
    ax.plot(budgets, expected_NPV[str(bound)].values(), marker = 'o', label='Expected NPV')
    ax.plot(budgets, actual_NPV[str(bound)].values(), marker = 'd', label='Actual NPV')
    ax.plot(budgets, optimal_NPV[str(bound)].values(), marker = 'x', label='"Optimal" NPV')
    ax.plot(budgets, expected_NPV_all_data[str(bound)].values(), marker = 'o', label='Expected NPV (all data)', linestyle='dashed')
    ax.plot(budgets, actual_NPV_all_data[str(bound)].values(), marker = 'd', label='Actual NPV (all data)', linestyle='dashed')
    ax.plot(budgets, actual_optimal_NPV[str(bound)].values(), marker = 'x', label='"Optimal" NPV (all data)', linestyle='dashed')
    ax.set_xlabel('Budget [M€]')
    ax.set_ylabel('NPV [M€]')
    ax.set_xlim([0, 2000])
    if bound == np.inf:
        bound = 'Unlimited'
    ax.set_title('Investment bound: {0} MW'.format(bound))
axs[1][1].legend(bbox_to_anchor=(-0.15, -0.3), loc='lower right', ncol=2)
plt.show()

print()