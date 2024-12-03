import gurobipy as gb
from network import Network
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class InvestmentPlanning(Network):
    
    def __init__(self, hours:int = 24, budget:float = 100, timelimit:float=100, carbontax:float=50, seed:int=42): # initialize class
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        
        self.T = hours
        self.offshore_flux = np.ones(hours)
        self.solar_flux = np.ones(hours)
        self.gas_flux = np.ones(hours)
        self.nuclear_flux = np.ones(hours)
        self.onshore_flux = np.ones(hours)
        self.carbontax = carbontax
        self.chosen_days = None

        if hours >= 24:
            assert hours % 24 == 0, "Hours must be a multiple of 24"
            days = hours // 24
            self.chosen_days = np.random.choice(range(365), days, replace=False)
            self.cf = {g: 1 for g in self.INVESTMENTS} # If typical days are used, set capacity factors to 1.

            self.offshore_flux = np.concatenate([self.offshore_hourly_2019[d*24: (d+1)*24].values for d in self.chosen_days])
            self.solar_flux = np.concatenate([self.solar_hourly_2019[d*24: (d+1)*24].values for d in self.chosen_days])
            self.onshore_flux = np.concatenate([self.onshore_hourly_2019[d*24: (d+1)*24].values for d in self.chosen_days])
            self.demand_profiles = np.concatenate([self.demand_hourly.iloc[d*24: (d+1)*24]['Demand'].values for d in self.chosen_days])
            self.TIMES = ['T{0}'.format(t) for t in range(1, days*24+1)]
            self.P_D = {} # Distribution of system demands
            for t, key in enumerate(self.TIMES):
                self.P_D[key] = dict(zip(self.DEMANDS, self.load_info['load_percent']/100 * self.demand_profiles[t]))
            #self.HOURS = ['T{0}'.format(t) for t in range(1, 24+1)]*days
        else:
            self.TIMES = ['T{0}'.format(t) for t in range(1, hours+1)]
            #self.HOURS = self.TIMES

        # Establish fluxes (primarily capping generation capacities of renewables)
        self.fluxes = {'Onshore Wind': self.onshore_flux,
                       'Offshore Wind': self.offshore_flux,
                       'Solar': self.solar_flux,
                       'Nuclear': self.nuclear_flux,
                       'Gas': self.gas_flux}
        
        # Define generation costs
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS
        self.C_G_offer = {g: self.C_G_offer[g] + (self.EF[g]*self.carbontax) for g in self.GENERATORS} # Variable costs in €/MWh incl. carbon tax
        self.C_I_offer = {g: self.v_OPEX[g] * 10**6 for g in self.INVESTMENTS} # Variable costs in €/MWh
        self.C_offer = {**self.C_G_offer, **self.C_I_offer, **self.C_W_offer} # Indexed by PRODUCTION_UNITS

        self.BUDGET = budget # set budget for capital costs in M€
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)

    def _add_lower_level_variables(self):
        # Define variables of lower level KKTs
        self.variables.p_g = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.p_d = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.lmd = {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='spot price at time {0}'.format(t)) for t in self.TIMES} # Hourly spot price (€/MWh)
        
        self.variables.mu_under    = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over     = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.sigma_over  = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}

        # Add binary auxiliary variables for non-convex constraints
        self.variables.q = {g: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='q_{0}_{1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.z = {g: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='z_{0}_{1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.y = {d: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='y_{0}_{1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.x = {d: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='x_{0}_{1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
    
    def _add_lower_level_constraints(self):
        M = max(self.P_G_max[g] for g in self.GENERATORS) # Big M for binary variables
        
        # KKT for lagrange objective derived wrt. generation variables
        self.constraints.gen_lagrange   = self.model.addConstrs((self.C_offer[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.PRODUCTION_UNITS for t in self.TIMES), name = "derived_lagrange_generators")
        #self.constraints.gen_lagrange_generators   = self.model.addConstrs((self.C_G_offer[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.GENERATORS for t in self.TIMES), name = "derived_lagrange_generators")
        #self.constraints.gen_lagrange_windturbines = self.model.addConstrs((self.C_W_offer[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.WINDTURBINES for t in self.TIMES), name = "derived_lagrange_windturbines")
        #self.constraints.gen_lagrange_investments  = self.model.addConstrs((self.C_I_offer[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.INVESTMENTS for t in self.TIMES), name = "derived_lagrange_investments")
        
        # KKT for lagrange objective derived wrt. demand variables
        self.constraints.dem_lagrange = self.model.addConstrs((-self.U_D[d] + self.variables.lmd[t] - self.variables.sigma_under[d][t] + self.variables.sigma_over[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "derived_lagrange_demand")
        
        # KKT for generation minimal production. Bi-linear are replaced by linearized constraints
        # self.constraints.gen_under = self.model.addConstrs((-self.variables.p_g[g][t] * self.variables.mu_under[g][t] == 0 for g in self.PRODUCTION_UNITS for t in self.TIMES), name = "gen_under")
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][t] <= self.variables.z[g][t] * self.P_G_max[g] for g in self.GENERATORS for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][t] <= self.variables.z[g][t] * self.P_W[g] for g in self.WINDTURBINES for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][t] <= self.variables.z[g][t] * M for g in self.INVESTMENTS for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_2 = self.model.addConstrs((self.variables.mu_under[g][t] <= M * (1 - self.variables.z[g][t]) for g in self.PRODUCTION_UNITS for t in self.TIMES), name = "gen_under_2") 

        # KKT for generation capacities. Bi-linear are replaced by linearized constraints
        # self.constraints.gen_upper_generators = self.model.addConstrs(((self.variables.p_g[g][t] - self.P_G_max[g]) * self.variables.mu_over[g][t] == 0 for g in self.GENERATORS for t in self.TIMES), name = "gen_upper_generators")
        self.constraints.gen_upper_generators_1 = self.model.addConstrs((self.variables.p_g[g][t] <= self.P_G_max[g] + M * self.variables.q[g][t] for g in self.GENERATORS for t in self.TIMES), name = "gen_upper_generators_1")
        self.constraints.gen_upper_generators_2 = self.model.addConstrs((self.P_G_max[g] - M * self.variables.q[g][t] <= self.variables.p_g[g][t] for g in self.GENERATORS for t in self.TIMES), name = "gen_upper_generators_2")
        self.constraints.gen_upper_generators_3 = self.model.addConstrs((self.variables.mu_over[g][t] <= M * (1 - self.variables.q[g][t]) for g in self.GENERATORS for t in self.TIMES), name = "gen_upper_generators_3")

        # self.constraints.gen_upper_windturbines = self.model.addConstrs(((self.variables.p_g[g][t] - self.P_W[t][g]) * self.variables.mu_over[g][t] == 0 for g in self.WINDTURBINES for t in self.TIMES), name = "gen_upper_windturbines")
        self.constraints.gen_upper_windturbines_1 = self.model.addConstrs((self.variables.p_g[g][t] <= self.P_W[t][g] + M * self.variables.q[g][t] for g in self.WINDTURBINES for t in self.TIMES), name = "gen_upper_windturbines_1")
        self.constraints.gen_upper_windturbines_2 = self.model.addConstrs((self.P_W[t][g] - M * self.variables.q[g][t] <= self.variables.p_g[g][t] for g in self.WINDTURBINES for t in self.TIMES), name = "gen_upper_windturbines_2")
        self.constraints.gen_upper_windturbines_3 = self.model.addConstrs((self.variables.mu_over[g][t] <= M * (1 - self.variables.q[g][t]) for g in self.WINDTURBINES for t in self.TIMES), name = "gen_upper_windturbines_3")

        # self.constraints.gen_upper_investments = self.model.addConstrs(((self.variables.p_g[g][t] - self.variables.P_investment[g] * self.fluxes[g][t_ix]) * self.variables.mu_over[g][t] == 0 for g in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments")
        self.constraints.gen_upper_investments_1 = self.model.addConstrs((self.variables.p_g[g][t] <= self.variables.P_investment[g] * self.fluxes[g][t_ix] + M * self.variables.q[g][t] for g in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments_1")
        self.constraints.gen_upper_investments_2 = self.model.addConstrs((self.variables.P_investment[g] * self.fluxes[g][t_ix] - M * self.variables.q[g][t] <= self.variables.p_g[g][t] for g in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments_2")
        self.constraints.gen_upper_investments_3 = self.model.addConstrs((self.variables.mu_over[g][t] <= M * (1 - self.variables.q[g][t]) for g in self.INVESTMENTS for t in self.TIMES), name = "gen_upper_investments_3")

        # KKT for demand constraints. Bi-linear are replaced by linearized constraints
        # self.constraints.dem_under = self.model.addConstrs((-self.variables.p_d[d][t] * self.variables.sigma_under[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_under_1 = self.model.addConstrs((self.variables.p_d[d][t] <= self.variables.y[d][t] * M for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_under_2 = self.model.addConstrs((self.variables.sigma_under[d][t] <= M * (1 - self.variables.y[d][t]) for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        
        
        # Redundant constraints
        # self.constraints.dem_upper = self.model.addConstrs(((self.variables.p_d[d][t] - self.P_D[t][d]) * self.variables.sigma_over[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_upper")
        # self.constraints.dem_upper_1 = self.model.addConstrs((self.variables.p_d[d][t] <= self.P_D[t][d] + M * self.variables.x[d][t] for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_1")
        self.constraints.dem_upper_2 = self.model.addConstrs((self.P_D[t][d] - M * self.variables.x[d][t] <= self.variables.p_d[d][t] for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_3")
        self.constraints.dem_upper_3 = self.model.addConstrs((self.variables.sigma_over[d][t] <= M * (1 - self.variables.x[d][t]) for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_2")
        
        
        # Generation capacity limits
        self.constraints.gen_cap_generators   = self.model.addConstrs((self.variables.p_g[g][t] <= self.P_G_max[g] for g in self.GENERATORS for t in self.TIMES), name = "gen_cap_generators")
        self.constraints.gen_cap_windturbines = self.model.addConstrs((self.variables.p_g[g][t] <= self.P_W[t][g] for g in self.WINDTURBINES for t in self.TIMES), name = "gen_cap_windturbines")
        self.constraints.gen_cap_investments  = self.model.addConstrs((self.variables.p_g[g][t] <= self.variables.P_investment[g] * self.fluxes[g][t_ix] for g in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES)), name = "gen_cap_investments")
        
        # Demand magnitude constraints
        self.constraints.dem_mag = self.model.addConstrs((self.variables.p_d[d][t] <= self.P_D[t][d] for d in self.DEMANDS for t in self.TIMES), name = "dem_mag")
        
        # KKT for balancing constraint
        self.constraints.balance = self.model.addConstrs((gb.quicksum(self.variables.p_g[g][t] for g in self.PRODUCTION_UNITS) - gb.quicksum(self.variables.p_d[d][t] for d in self.DEMANDS) == 0  for t in self.TIMES), name = "balance")

    def build_model(self):
        self.model = gb.Model(name='Investment Planning')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        # Investment in generation technologies (in MW)
        self.variables.P_investment = {g : self.model.addVar(lb=0, ub=GRB.INFINITY, name='investment in {0}'.format(g)) for g in self.INVESTMENTS}
        
        # Get lower level variables
        self._add_lower_level_variables()

        self.model.update()

        """ Initialize objective to maximize NPV [M€] """
        # Define costs (annualized capital costs + fixed and variable operational costs)
        costs = gb.quicksum( self.variables.P_investment[g] * (self.AF[g] * self.CAPEX[g] + self.f_OPEX[g])
                            + 8760/self.T * gb.quicksum(self.v_OPEX[g] * self.variables.p_g[g][t] for t in self.TIMES)
                            for g in self.INVESTMENTS)
        # Define revenue (sum of generation revenues) [M€]
        revenue = (8760 / self.T / 10**6) * gb.quicksum(self.cf[g] * 
                                            self.variables.lmd[t] * self.variables.p_g[g][t]
                                            for g in self.INVESTMENTS for t in self.TIMES)
        
        # Define NPV, including magic constant
        npv = revenue - costs

        # Set objective
        self.model.setObjective(npv, gb.GRB.MAXIMIZE)

        self.model.update()

        """ Initialize constraints """
        # Budget constraints
        self.constraints.budget = self.model.addConstr(gb.quicksum(
                                    self.variables.P_investment[g] * self.CAPEX[g] for g in self.INVESTMENTS)
                                    <= self.BUDGET, name='budget')
        
        self._add_lower_level_constraints()

        # Set non-convex objective
        self.model.Params.NonConvex = 2
        self.model.update()

    def run(self):
        self.model.optimize()
        self._save_data()
    
    def _calculate_capture_prices(self):
        # Calculate capture price
        self.data.capture_prices = {
            g : (np.sum(self.data.lambda_[t] * self.data.investment_dispatch_values[t][g] for t in self.TIMES) /
                    np.sum(self.data.investment_dispatch_values[t][g] for t in self.TIMES)) if self.data.investment_values[g] > 0 else None
            for g in self.INVESTMENTS}

    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save investment values
        self.data.investment_values = {g : self.variables.P_investment[g].x for g in self.INVESTMENTS}
        self.data.capacities = {t :
                                {**{g : self.data.investment_values[g]*self.fluxes[g][t_ix] for g in self.INVESTMENTS},
                                **{g : self.P_G_max[g] for g in self.GENERATORS},
                                **{g : self.P_W[t][g] for g in self.WINDTURBINES}}
                                for t_ix, t in enumerate(self.TIMES)}
        
        # Save generation dispatch values
        self.data.investment_dispatch_values = {t : {g : self.variables.p_g[g][t].x for g in self.INVESTMENTS} for t in self.TIMES}
        self.data.generator_dispatch_values = {t : {g : self.variables.p_g[g][t].x for g in self.GENERATORS} for t in self.TIMES}
        self.data.windturbine_dispatch_values = {t : {g : self.variables.p_g[g][t].x for g in self.WINDTURBINES} for t in self.TIMES}
        self.data.all_dispatch_values = {t : {g : self.variables.p_g[g][t].x for g in self.PRODUCTION_UNITS} for t in self.TIMES}

        # Save demand dispatch values
        self.data.demand_dispatch_values = {t : {d: self.variables.p_d[d][t].x for d in self.DEMANDS} for t in self.TIMES}
        
        # Save uniform prices lambda
        self.data.lambda_ = {t : self.variables.lmd[t].x for t in self.TIMES}

        self._calculate_capture_prices()
    
    def _get_demand_curve_info(self, T: str):
        bids = pd.Series(self.U_D).sort_values(ascending=False)
        dispatch = pd.Series(self.data.demand_dispatch_values[T]).loc[bids.index]
        demand = pd.Series(self.P_D[T]).loc[bids.index]
        dispatch_cumulative = dispatch.copy()
        demand_cumulative = demand.copy()
        for i in range(1, len(dispatch_cumulative)):
            dispatch_cumulative.iloc[i] += dispatch_cumulative.iloc[i-1]
            demand_cumulative.iloc[i] += demand_cumulative.iloc[i-1]
        
        dispatch_cumulative = pd.concat([pd.Series([0]), dispatch_cumulative])
        demand_cumulative = pd.concat([pd.Series([0]), demand_cumulative])
        
        bids = pd.concat([pd.Series([bids.iloc[0]]), bids])
        
        return dispatch_cumulative, bids, demand_cumulative
    
    def _get_supply_curve_info(self, T: str):
        offers = pd.Series(self.C_offer).sort_values(ascending=True)
        dispatch = pd.Series(self.data.all_dispatch_values[T]).loc[offers.index]
        capacity = pd.Series(self.data.capacities[T]).loc[offers.index]
        dispatch_cumulative = dispatch.cumsum()
        capacity_cumulative = capacity.cumsum()
        # for i in range(1, len(dispatch_cumulative)):
        #     dispatch_cumulative.iloc[i] += dispatch_cumulative.iloc[i-1]
        #     capacity_cumulative.iloc[i] += capacity_cumulative.iloc[i-1]
        dispatch_cumulative = pd.concat([pd.Series([0]), dispatch_cumulative])
        capacity_cumulative = pd.concat([pd.Series([0]), capacity_cumulative])
        offers = pd.concat([pd.Series(offers.iloc[0]), offers])
        
        return dispatch_cumulative, offers, capacity_cumulative
        
    def plot_supply_demand_curve(self, T: str):
        plt.figure(figsize=(10, 6))
        dispatch_cumulative, offers, capacity_cumulative = self._get_supply_curve_info(T)
        bought_cumulative, bids, demand_cumulative = self._get_demand_curve_info(T)
        plt.step(np.array(capacity_cumulative), np.array(offers), label='Supply', )
        plt.step(np.array(demand_cumulative), np.array(bids), label='Demand', )
        #plt.step(np.array(dispatch_cumulative), np.array(offers), label='Supply dispatched')
        #plt.step(np.array(bought_cumulative), np.array(bids), label='Demand met')
        plt.xlabel('Quantity [MWh]')
        plt.ylabel('Power Price [€/MWh]')
        plt.title('Supply/Demand curve for time {0}'.format(T))
        plt.xlim(0, max(capacity_cumulative.max(), demand_cumulative.max()))
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_clearing_prices(self):
        data = [ip.data.lambda_[t] for t in ip.TIMES]
        plt.figure(figsize=(10, 6))

        # Plot the data
        plt.plot(data, label='Power Price', linewidth=2)

        # Add labels and title
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power Price [€/MWh]', fontsize=12)
        plt.title('Clearing Prices', fontsize=14)

        # Add only horizontal grid lines
        plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

        # Improve layout with tight layout and legend
        plt.tight_layout()
        plt.legend()

        # Show the plot
        plt.show()

    def display_results(self):
        print('Maximal NPV: \t{0} M€\n'.format(round(self.data.objective_value,2)))
        print('Investment Capacities:')
        for key, value in self.data.investment_values.items():
            print(f"{key}: \t\t{round(value,2)} MW")
            print(f"Capital cost: \t\t{round(value*self.CAPEX[key],2)} M€\n")



if __name__ == '__main__':
    # Carbon TAX price: https://www.statista.com/statistics/1322214/carbon-prices-european-union-emission-trading-scheme/
    # Carbon TAX price: https://www.eex.com/en/market-data/emission-allowances/eua-auction-results
    
    # Initialize investment planning model
    ip = InvestmentPlanning(hours=20*24, budget=1000, timelimit=600, carbontax=60, seed=38)

    # Build model
    ip.build_model()
    # Run optimization
    ip.run()
    # Display results
    ip.display_results()

    ip.plot_supply_demand_curve('T1')
