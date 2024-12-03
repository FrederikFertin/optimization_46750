#%%
import gurobipy as gb
from network import Network
from gurobipy import GRB
import numpy as np
import random
import pandas as pd
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class nodal_clearing(Network):
    
    def __init__(self, hours:int = 24, timelimit:float = 100, carbontax:float = 50, seed:int = 42, P_investment = None): # initialize class
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        
        self.T = hours
        self.carbontax = carbontax
        self.chosen_days = None
        self.root_node = 'N1'
        self.other_nodes = self.NODES.copy()
        self.other_nodes.remove(self.root_node)

        self.P_investment = P_investment
        if P_investment is None:
            self.P_investment = {i: {n: 0 for n in self.node_I[i]} for i in self.INVESTMENTS}

        self._initialize_fluxes(hours)
        self._initialize_times_and_demands(hours, timelimit)

    def _initialize_fluxes(self, hours):
        self.offshore_flux = np.ones(hours)
        self.solar_flux    = np.ones(hours)
        self.gas_flux      = np.ones(hours)
        self.nuclear_flux  = np.ones(hours)
        self.onshore_flux  = np.ones(hours)

        

    def _initialize_times_and_demands(self, hours, timelimit):
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
        else:
            self.TIMES = ['T{0}'.format(t) for t in range(1, hours+1)]        
        
        # Define generation costs
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS
        self.node_production = {**self.node_G, **self.node_I, **self.node_W}
        self.C_G_offer = {g: self.C_G_offer[g] + (self.EF[g]*self.carbontax) for g in self.GENERATORS} # Variable costs in €/MWh incl. carbon tax
        self.C_I_offer = {g: self.v_OPEX[g] * 10**6 for g in self.INVESTMENTS} # Variable costs in €/MWh
        self.C_offer = {**self.C_G_offer, **self.C_I_offer, **self.C_W_offer} # Indexed by PRODUCTION_UNITS

        # Establish fluxes (primarily capping generation capacities of renewables)
        self.fluxes = {'Onshore Wind'  : self.onshore_flux,
                       'Offshore Wind' : self.offshore_flux,
                       'Solar'         : self.solar_flux,
                       'Nuclear'       : self.nuclear_flux,
                       'Gas'           : self.gas_flux}

        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)


    def build_model(self):
        self.model = gb.Model(name='Nodal clearing')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        self.variables.p_g          = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d          = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta        = {n: {t:       self.model.addVar(lb=-np.pi,        ub=np.pi,        name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.NODES}
        self.variables.flow         = {l: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='flow_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

        """ Initialize objective function """
        self.model.setObjective(gb.quicksum(self.U_D[d] * self.variables.p_d[d][n][t] for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES)
                               -gb.quicksum(self.C_offer[g] * self.variables.p_g[g][n][t] for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES)
                                , gb.GRB.MAXIMIZE)
        self.model.update()

        """ Initialize constraints """
        # Define flow auxiliary variables
        self.constraints.flow       = self.model.addConstrs((self.variables.flow[l][t] == self.L_susceptance[l] * (self.variables.theta[self.node_L_from[l]][t] - self.variables.theta[self.node_L_to[l]][t])
                                                            for l in self.LINES for t in self.TIMES), name = "flow")
        # Generation capacity limits:
        self.constraints.p_gen_cap  = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g]
                                                            for g in self.GENERATORS for t in self.TIMES for n in self.node_G[g]), name = "gen_cap_generators")
        self.constraints.p_wt_cap   = self.model.addConstrs((self.variables.p_g[w][n][t] <= self.P_W[t][w]
                                                            for w in self.WINDTURBINES for t in self.TIMES for n in self.node_W[w]), name = "gen_cap_windturbines")
        self.constraints.p_inv_cap  = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.P_investment[i][n] * self.fluxes[i][t_ix]
                                                            for i in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES) for n in self.node_I[i]), name = "gen_cap_investments")
        # Demand magnitude constraints:
        self.constraints.demand_cap = self.model.addConstrs((self.variables.p_d[d][n][t] <= self.P_D[t][d]
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_mag")
        # Balancing constraint for each node:
        self.constraints.balance    = self.model.addConstrs(( gb.quicksum(self.variables.p_d[d][n][t]   for d in self.map_d[n])
                                                            - gb.quicksum(self.variables.p_g[g][n][t]   for g in self.map_g[n])
                                                            - gb.quicksum(self.variables.p_g[w][n][t]   for w in self.map_w[n])
                                                            - gb.quicksum(self.variables.p_g[i][n][t]   for i in self.map_i[n])
                                                            + gb.quicksum(self.variables.flow[l][t]     for l in self.map_from[n])
                                                            - gb.quicksum(self.variables.flow[l][t]     for l in self.map_to[n]) == 0
                                                            for n in self.NODES for t in self.TIMES), name = "balance")
        # Line capacity constraints:
        self.constraints.line_l_cap = self.model.addConstrs((-self.variables.flow[l][t] <= self.L_cap[l]
                                                            for l in self.LINES for t in self.TIMES), name = "line_cap_lower")
        self.constraints.line_u_cap = self.model.addConstrs((self.variables.flow[l][t]  <= self.L_cap[l]
                                                            for l in self.LINES for t in self.TIMES), name = "line_cap_lower")
        # Reference voltage angle:
        self.constraints.ref_angle  = self.model.addConstrs((  self.variables.theta[self.root_node][t] == 0
                                                            for t in self.TIMES), name = "ref_angle")
        self.constraints.angle_u    = self.model.addConstrs((  self.variables.theta[n][t] <= np.pi
                                                            for n in self.NODES for t in self.TIMES), name = "angle_upper_limit") # Can these limits be imposed in the definition of the variable?
        self.constraints.angle_l    = self.model.addConstrs((- self.variables.theta[n][t] <= np.pi
                                                            for n in self.NODES for t in self.TIMES), name = "angle_lower_limit")

        self.model.update()

    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save dispatch values
        self.data.generator_dispatch_values = {g : {n : {t : self.variables.p_g[g][n][t].x for t in self.TIMES} for n in self.node_G[g]} for g in self.GENERATORS}

        # Save generation dispatch values
        self.data.investment_dispatch_values = {g : {n : {t : self.variables.p_g[g][n][t].x for t in self.TIMES} for n in self.node_I[g]} for g in self.INVESTMENTS}
        
        # Save uniform prices lambda
        self.data.lambda_ = {n : {t : self.constraints.balance[n,t].pi for t in self.TIMES} for n in self.NODES}

        # Save voltage angles
        self.data.theta = {n : {t : self.variables.theta[n][t].x for t in self.TIMES} for n in self.NODES}

        # Save line flows
        self.data.flow = {l : {t : self.variables.flow[l][t].x for t in self.TIMES} for l in self.LINES}
        self.data.l_cap_l_dual = {l : {t : self.constraints.line_l_cap[l,t].pi for t in self.TIMES} for l in self.LINES}
        self.data.l_cap_u_dual = {l : {t : self.constraints.line_u_cap[l,t].pi for t in self.TIMES} for l in self.LINES}

        self.costs = np.sum(self.P_investment[g][n] * (self.AF[g] * self.CAPEX[g] + self.f_OPEX[g])
                            + 8760/self.T * np.sum(self.v_OPEX[g] * self.data.investment_dispatch_values[g][n][t] for t in self.TIMES)
                            for g in self.INVESTMENTS for n in self.node_I[g])
        # Define revenue (sum of generation revenues) [M€]
        self.revenue = (8760 / self.T / 10**6) * np.sum(self.cf[g] * 
                                            self.data.lambda_[n][t] * self.data.investment_dispatch_values[g][n][t]
                                            for g in self.INVESTMENTS for t in self.TIMES for n in self.node_I[g])
        
        # Define NPV
        self.data.npv = self.revenue - self.costs

    def run(self):
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        self._save_data()
    
    def display_results(self):
        print('Actual NPV: \t{0} M€\n'.format(round(self.data.npv,2)))
        

    def plot_prices(self):     
        # Plot boxplots of price distribution in each node
        prices = pd.DataFrame(self.data.lambda_)
        prices.plot(kind='box', figsize=(10,5))
        plt.title('Price distribution at each node')
        plt.ylabel('Price [€/MWh]')
        plt.show()


class InvestmentPlanning(Network):
    
    def __init__(self, hours:int = 24, budget:float = 100, timelimit:float=100, carbontax:float=50, seed:int=42, lmd:dict = None): # initialize class
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        
        self.lmd = lmd
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
        else:
            self.TIMES = ['T{0}'.format(t) for t in range(1, hours+1)]

        # Establish fluxes (primarily capping generation capacities of renewables)
        self.fluxes = {'Onshore Wind': self.onshore_flux,
                       'Offshore Wind': self.offshore_flux,
                       'Solar': self.solar_flux,
                       'Nuclear': self.nuclear_flux,
                       'Gas': self.gas_flux}
        
        self.BUDGET = budget # set budget for capital costs in M€
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)

    def build_model(self):
        self.model = gb.Model(name='Investment Planning')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        # Investment in generation technologies (in MW)
        self.variables.P_investment = {g : {n :     self.model.addVar(lb=0, ub=GRB.INFINITY, name='investment in {0}'.format(g)) for n in self.node_I[g]} for g in self.INVESTMENTS}
        self.variables.p_g =          {g : {n : {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for n in self.node_I[g]} for g in self.INVESTMENTS}
        self.model.update()

        """ Initialize objective to maximize NPV [M€] """
        # Define costs (annualized capital costs + fixed and variable operational costs)
        self.costs = gb.quicksum(self.variables.P_investment[g][n] * (self.AF[g] * self.CAPEX[g] + self.f_OPEX[g])
                            + 8760/self.T * gb.quicksum(self.v_OPEX[g] * self.variables.p_g[g][n][t] for t in self.TIMES)
                            for g in self.INVESTMENTS for n in self.node_I[g])
        # Define revenue (sum of generation revenues) [M€]
        self.revenue = (8760 / self.T / 10**6) * gb.quicksum(self.cf[g] * 
                                            self.lmd[n][t] * self.variables.p_g[g][n][t]
                                            for g in self.INVESTMENTS for t in self.TIMES for n in self.node_I[g])
        
        # Define NPV
        npv = self.revenue - self.costs

        # Set objective
        self.model.setObjective(npv, gb.GRB.MAXIMIZE)

        self.model.update()

        """ Initialize constraints """
        # Budget constraints
        self.constraints.budget = self.model.addConstr(gb.quicksum(
                                    self.variables.P_investment[g][n] * self.CAPEX[g] for g in self.INVESTMENTS for n in self.node_I[g])
                                    <= self.BUDGET, name='budget')
        # Generation capacity limits
        self.constraints.gen_cap_investments  = self.model.addConstrs((self.variables.p_g[g][n][t] 
                                                                       <= self.variables.P_investment[g][n] * self.fluxes[g][t_ix]
                                                                         for g in self.INVESTMENTS for n in self.node_I[g] for t_ix, t in enumerate(self.TIMES)), name = "gen_cap_investments")

        # Set non-convex objective
        self.model.update()


    def run(self):
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        self._save_data()
    
    def _calculate_capture_prices(self):
        # Calculate capture price
        self.data.capture_prices = {
            g : {n : (np.sum(np.fromiter((self.lmd[n][t] * self.data.investment_dispatch_values[g][n][t] for t in self.TIMES), float)) /
                    np.sum(np.fromiter((self.data.investment_dispatch_values[g][n][t] for t in self.TIMES), float))) if self.data.investment_values[g][n] > 0 else None
            for n in self.node_I[g]} for g in self.INVESTMENTS}

    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save investment values
        self.data.investment_values = {g : {n : self.variables.P_investment[g][n].x for n in self.node_I[g]} for g in self.INVESTMENTS}
        self.data.capacities = {t :
                                {g : self.data.investment_values[g][n]*self.fluxes[g][t_ix] for g in self.INVESTMENTS for n in self.node_I[g]}
                                for t_ix, t in enumerate(self.TIMES)}
        
        # Save generation dispatch values
        self.data.investment_dispatch_values = {g : { n : {t : self.variables.p_g[g][n][t].x for t in self.TIMES} for n in self.node_I[g]} for g in self.INVESTMENTS}
        
        self._calculate_capture_prices()

    def display_results(self):
        print('Expected NPV: \t{0} M€\n'.format(round(self.data.objective_value,2)))
        print('Investment Capacities:')
        for g_type, nodal_investments in self.data.investment_values.items():
            capex = 0
            for node, investment_size in nodal_investments.items():
                capex += investment_size*self.CAPEX[g_type]
                if investment_size > 0:
                    print(f"{g_type} at {node}: \t{round(investment_size,2)} MW")
            if capex > 0: 
                print(f"Capital cost for {g_type}: \t\t{round(capex,2)} M€\n")

 
#%%
if __name__ == '__main__':
    # Model parameters
    hours = 180*24
    timelimit = 600
    carbontax = 60
    seed = 38
    expected_NPV = []
    actual_NPV = []

    # Create nodal clearing instance without new investments for a price forecast
    nc_org = nodal_clearing(hours=hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
    nc_org.build_model()
    nc_org.run()
    # nc.plot_prices()
    price_forcast = nc_org.data.lambda_
    p_forecast = pd.DataFrame(price_forcast)


# %%
for budget in [1000]:
    ip = InvestmentPlanning(hours=hours, budget = budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd=price_forcast)
    ip.build_model()
    ip.run()
    ip.display_results()
    investments=ip.data.investment_values
    expected_NPV.append(ip.data.objective_value)

    # Create nodal clearing instance with new investments
    nc = nodal_clearing(hours=hours, timelimit=timelimit, carbontax=carbontax, seed=seed, P_investment=investments)
    nc.build_model()
    nc.run()
    nc.display_results()
    actual_NPV.append(nc.data.npv)

# %%
plt.plot(np.logspace(0, 4, 5), expected_NPV, label='Expected NPV')
plt.plot(np.logspace(0, 4, 5), actual_NPV, label='Actual NPV')
plt.xscale('log')
plt.xlabel('Budget [M€]')
plt.yscale('log')
plt.ylabel('NPV [M€]')
plt.legend()
plt.show()
# %%
