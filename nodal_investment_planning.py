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

class InvestmentPlanning(Network):
    
    def __init__(self, hours:int = 24, budget:float = 100, timelimit:float=100, carbontax:float=50, seed:int=42): # initialize class
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        
        self.T = hours
        self.carbontax = carbontax
        self.chosen_days = None

        self._initialize_fluxes(hours)
        self._initialize_times_and_demands(hours)
        self._initialize_costs_and_budget(budget, timelimit)

    def _initialize_fluxes(self, hours):
        self.offshore_flux = np.ones(hours)
        self.solar_flux = np.ones(hours)
        self.gas_flux = np.ones(hours)
        self.nuclear_flux = np.ones(hours)
        self.onshore_flux = np.ones(hours)

    def _initialize_times_and_demands(self, hours):
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

    def _initialize_costs_and_budget(self, budget, timelimit):
        # Establish fluxes (primarily capping generation capacities of renewables)
        self.fluxes = {'Onshore Wind': self.onshore_flux,
                       'Offshore Wind': self.offshore_flux,
                       'Solar': self.solar_flux,
                       'Nuclear': self.nuclear_flux,
                       'Gas': self.gas_flux}
        
        # Define generation costs
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS
        self.node_production = {**self.node_G, **self.node_I, **self.node_W}
        self.C_G_offer = {g: self.C_G_offer[g] + (self.EF[g]*self.carbontax) for g in self.GENERATORS} # Variable costs in €/MWh incl. carbon tax
        self.C_I_offer = {g: self.v_OPEX[g] * 10**6 for g in self.INVESTMENTS} # Variable costs in €/MWh
        self.C_offer = {**self.C_G_offer, **self.C_I_offer, **self.C_W_offer} # Indexed by PRODUCTION_UNITS

        self.BUDGET = budget # set budget for capital costs in M€
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)

    def _add_lower_level_variables(self):
        # Define primal variables of lower level KKTs
        self.variables.p_g   = {g: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d   = {d: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta = {n: {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.NODES}
        self.variables.flow = {l: {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='flow_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

        # Define dual variables of lower level KKTs
        self.variables.lmd   = {n: {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='spot price at time {0} in node {1}'.format(t,n)) for t in self.TIMES} for n in self.NODES} # Hourly spot price (€/MWh)
        self.variables.mu_under    = {g: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over     = {g: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under = {d: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.sigma_over  = {d: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.rho_over    = {n: {m: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub between node {0} and node {1} at time {2}'.format(n, m, t)) for t in self.TIMES} for m, l in self.map_n[n].items()} for n in self.NODES}
        self.variables.rho_under   = {n: {m: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb between line {0} and line {1} at time {2}'.format(n, m, t)) for t in self.TIMES} for m, l in self.map_n[n].items()} for n in self.NODES}
        self.variables.nu = {'N1': {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Dual for reference angle at time {0}'.format(t)) for t in self.TIMES}}
        # "as rho_over is duplicated from n to m and m to n meaning both directions of DC power flow are considered"

        # Add binary auxiliary variables for bi-linear constraints
        self.variables.b1 = {g: {n: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b1_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.b2 = {g: {n: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b2_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.b3 = {d: {n: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b3_{0}_{1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.b4 = {d: {n: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b4_{0}_{1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.b5 = {n: {m: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b5_{0}_{1}_{2}'.format(n, m, t)) for t in self.TIMES} for m, l in self.map_n[n].items()} for n in self.NODES}
        self.variables.b6 = {n: {m: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b6_{0}_{1}_{2}'.format(n, m, t)) for t in self.TIMES} for m, l in self.map_n[n].items()} for n in self.NODES}
        #self.variables.b7 = {'N1': {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b7_{0}'.format(t)) for t in self.TIMES}}


    def _add_lower_level_constraints(self):
        M = max(self.P_G_max[g] for g in self.GENERATORS) # Big M for binary variables

        self.constraints.flow = self.model.addConstrs((self.variables.flow[l][t] == self.L_susceptance[l] * (self.variables.theta[self.node_L_from[l]][t] - self.variables.theta[self.node_L_to[l]][t]) for l in self.LINES for t in self.TIMES), name = "flow")

        # KKT for lagrange objective derived wrt. generation variables
        self.constraints.gen_lagrange = self.model.addConstrs((self.C_offer[g] - self.variables.lmd[n][t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES), name = "derived_lagrange_generators")
        #self.constraints.gen_lagrange_generators   = self.model.addConstrs((self.C_G_offer[g] - self.variables.lmd[n][t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "derived_lagrange_generators")
        #self.constraints.gen_lagrange_windturbines = self.model.addConstrs((self.v_OPEX['Offshore Wind'] - self.variables.lmd[t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "derived_lagrange_windturbines")
        #self.constraints.gen_lagrange_investments  = self.model.addConstrs((self.v_OPEX[g] - self.variables.lmd[n][t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.INVESTMENTS for n in self.NODES for t in self.TIMES), name = "derived_lagrange_investments")
        
        # KKT for lagrange objective derived wrt. demand variables
        self.constraints.dem_lagrange = self.model.addConstrs((-self.U_D[d] + self.variables.lmd[n][t] - self.variables.sigma_under[d][n][t] + self.variables.sigma_over[d][n][t] == 0 for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "derived_lagrange_demand")
        
        # KKT for lagrange objective derived wrt. line flow variables
        self.constraints.line_lagrange = self.model.addConstrs((self.L_susceptance[l] * (self.variables.lmd[n][t] + self.variables.rho_under[n][m][t] - self.variables.rho_over[n][m][t]) - (self.variables.nu[n][t] if n == 'N1' else 0) == 0 for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "derived_lagrange_line")
        #self.constraints.line_lagrange = self.model.addConstrs((self.L_susceptance[l] * (self.variables.lmd[n][t] - self.variables.rho_under[m][n][t] + self.variables.rho_over[m][n][t]) - (self.variables.nu[m][t] if m == 'N1' else 0) == 0 for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "derived_lagrange_line")

        # KKT for generation minimal production. Bi-linear are replaced by linearized constraints
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.P_G_max[g] for g in self.GENERATORS for n in self.node_G[g] for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_2 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.P_W[t][g] for g in self.WINDTURBINES for n in self.node_W[g] for t in self.TIMES), name = "gen_under_2")
        self.constraints.gen_under_3 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.variables.P_investment[g][n] * self.fluxes[g][t_ix] for g in self.INVESTMENTS for n in self.node_I[g] for t_ix, t in enumerate(self.TIMES)), name = "gen_under_3")
        self.constraints.gen_under_4 = self.model.addConstrs((self.variables.mu_under[g][n][t] <= M * (1 - self.variables.b1[g][n][t]) for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES), name = "gen_under_4") 

        # KKT for generation capacities. Bi-linear are replaced by linearized constraints
        #Redundant: #self.constraints.gen_upper_generators_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g] + M * self.variables.b2[g][n][t] for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "gen_upper_generators_1")
        self.constraints.gen_upper_generators_2 = self.model.addConstrs((self.P_G_max[g] - M * self.variables.b2[g][n][t] <= self.variables.p_g[g][n][t] for g in self.GENERATORS for n in self.node_G[g] for t in self.TIMES), name = "gen_upper_generators_2")
        self.constraints.gen_upper_generators_3 = self.model.addConstrs((self.variables.mu_over[g][n][t] <= M * (1 - self.variables.b2[g][n][t]) for g in self.GENERATORS for n in self.node_G[g] for t in self.TIMES), name = "gen_upper_generators_3")

        #Redundant: #self.constraints.gen_upper_windturbines_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_W[t][g] + M * self.variables.b2[g][n][t] for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "gen_upper_windturbines_1")
        self.constraints.gen_upper_windturbines_2 = self.model.addConstrs((self.P_W[t][g] - M * self.variables.b2[g][n][t] <= self.variables.p_g[g][n][t] for g in self.WINDTURBINES for n in self.node_W[g] for t in self.TIMES), name = "gen_upper_windturbines_2")
        self.constraints.gen_upper_windturbines_3 = self.model.addConstrs((self.variables.mu_over[g][n][t] <= M * (1 - self.variables.b2[g][n][t]) for g in self.WINDTURBINES for n in self.node_W[g] for t in self.TIMES), name = "gen_upper_windturbines_3")

        #Redundant: #self.constraints.gen_upper_investments_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.P_investment[g][n] * self.fluxes[g][t_ix] + M * self.variables.b2[g][n][t] for g in self.INVESTMENTS for n in self.NODES for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments_1")
        self.constraints.gen_upper_investments_2 = self.model.addConstrs((self.variables.P_investment[g][n] * self.fluxes[g][t_ix] - M * self.variables.b2[g][n][t] <= self.variables.p_g[g][n][t] for g in self.INVESTMENTS for n in self.node_I[g] for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments_2")
        self.constraints.gen_upper_investments_3 = self.model.addConstrs((self.variables.mu_over[g][n][t] <= M * (1 - self.variables.b2[g][n][t]) for g in self.INVESTMENTS for n in self.node_I[g] for t in self.TIMES), name = "gen_upper_investments_3")

        # KKT for demand constraints. Bi-linear are replaced by linearized constraints
        # self.constraints.dem_under = self.model.addConstrs((-self.variables.p_d[d][t] * self.variables.sigma_under[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_under_1 = self.model.addConstrs((self.variables.p_d[d][n][t] <= self.variables.b3[d][n][t] * self.P_D[t][d] for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_under_1")
        self.constraints.dem_under_2 = self.model.addConstrs((self.variables.sigma_under[d][n][t] <= M * (1 - self.variables.b3[d][n][t]) * M for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_under_2")
        
        # self.constraints.dem_upper_1 = self.model.addConstrs((self.variables.p_d[d][t] <= self.P_D[t][d] + M * self.variables.x[d][t] for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_1")
        self.constraints.dem_upper_2 = self.model.addConstrs((self.P_D[t][d] - M * self.variables.b4[d][n][t] <= self.variables.p_d[d][n][t] for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_upper_2")
        self.constraints.dem_upper_3 = self.model.addConstrs((self.variables.sigma_over[d][n][t] <= M * (1 - self.variables.b4[d][n][t]) for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_upper_3")
        
        # KKT for line flow constraints. Bi-linear are replaced by linearized constraints
        self.constraints.line_over_1 = self.model.addConstrs((self.variables.rho_over[n][m][t] <= M * (1 - self.variables.b5[n][m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_under_1")
        self.constraints.line_over_2 = self.model.addConstrs((-self.L_cap[l]/self.L_susceptance[l] - self.variables.b5[n][m][t] * M <= self.variables.theta[n][t] - self.variables.theta[m][t] for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_under_2")
        self.constraints.line_over_3 = self.model.addConstrs((self.variables.theta[n][t] - self.variables.theta[m][t] <= -self.L_cap[l]/self.L_susceptance[l] + self.variables.b5[n][m][t] * M for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_under_3")
        
        self.constraints.line_under_1 = self.model.addConstrs((self.variables.rho_under[n][m][t] <= M * (1 - self.variables.b6[n][m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "line_over_1")
        self.constraints.line_under_2 = self.model.addConstrs((self.L_cap[l]/self.L_susceptance[l] - self.variables.b6[n][m][t] * M <= self.variables.theta[n][t] - self.variables.theta[m][t] for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_over_2")
        self.constraints.line_under_3 = self.model.addConstrs((self.variables.theta[n][t] - self.variables.theta[m][t] <= self.L_cap[l]/self.L_susceptance[l] + self.variables.b6[n][m][t] * M for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_over_3")    

        ### Primal constraints ###
        # Generation capacity limits
        self.constraints.gen_cap_generators   = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g] for g in self.GENERATORS for t in self.TIMES for n in self.node_G[g]), name = "gen_cap_generators")
        self.constraints.gen_cap_windturbines = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_W[t][g] for g in self.WINDTURBINES for t in self.TIMES for n in self.node_W[g]), name = "gen_cap_windturbines")
        self.constraints.gen_cap_investments  = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.variables.P_investment[i][n] * self.fluxes[i][t_ix] for i in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES) for n in self.node_I[i]), name = "gen_cap_investments")

        # Demand magnitude constraints
        self.constraints.dem_mag = self.model.addConstrs((self.variables.p_d[d][n][t] <= self.P_D[t][d] for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_mag")
        
        # Balancing constraint
        self.constraints.balance = self.model.addConstrs((gb.quicksum(self.variables.p_g[g][n][t] for g in self.map_g[n])
                                                        + gb.quicksum(self.variables.p_g[w][n][t] for w in self.map_w[n])
                                                        + gb.quicksum(self.variables.p_g[i][n][t] for i in self.map_i[n])
                                                        - gb.quicksum(self.variables.p_d[d][n][t] for d in self.map_d[n])
                                                        + gb.quicksum(self.L_susceptance[l] * (self.variables.theta[n][t] - self.variables.theta[m][t]) for m, l in self.map_n[n].items()) == 0  for t in self.TIMES for n in self.NODES), name = "balance")

        # Line capacity constraints
        self.constraints.line_cap_lower = self.model.addConstrs((- self.L_cap[l] <= self.L_susceptance[l] * (self.variables.theta[n][t] - self.variables.theta[m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "line_cap_lower")
        #Redundant: #self.constraints.line_cap_upper = self.model.addConstrs((self.L_cap[l] >= self.L_susceptance[l] * (self.variables.theta[n][t] - self.variables.theta[m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "line_cap_upper")
        
        # Reference voltage angle
        self.constraints.ref_angle = self.model.addConstrs((self.variables.theta['N1'][t] == 0 for t in self.TIMES), name = "ref_angle")

    def build_model(self):
        self.model = gb.Model(name='Investment Planning')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        # Investment in generation technologies (in MW)
        self.variables.P_investment = {i : {n : self.model.addVar(lb=0, ub=GRB.INFINITY, name='investment in tech {0} at node {1}'.format(i, n)) for n in self.node_I[i]} for i in self.INVESTMENTS}
        
        # Get lower level variables
        self._add_lower_level_variables()

        self.model.update()

        """ Initialize objective to maximize NPV [M€] """
        # Define costs (annualized capital costs + fixed and variable operational costs)
        costs = gb.quicksum( self.variables.P_investment[i][n] * (self.AF[i] * self.CAPEX[i] + self.f_OPEX[i])
                            + 8760/self.T * gb.quicksum(self.v_OPEX[i] * self.variables.p_g[i][n][t] for t in self.TIMES)
                            for i in self.INVESTMENTS for n in self.node_I[i])
        # Define revenue (sum of generation revenues) [M€]
        revenue = (8760 / self.T / 10**6) * gb.quicksum(self.cf[i] * 
                                            self.variables.lmd[n][t] * self.variables.p_g[i][n][t]
                                            for i in self.INVESTMENTS for t in self.TIMES for n in self.node_I[i])
        
        # Define NPV, including magic constant
        npv = revenue - costs

        # Set objective
        self.model.setObjective(npv, gb.GRB.MAXIMIZE)

        self.model.update()

        """ Initialize constraints """
        # Budget constraints
        self.constraints.budget = self.model.addConstr(gb.quicksum(
                                    self.variables.P_investment[i][n] * self.CAPEX[i] for i in self.INVESTMENTS for n in self.node_I[i])
                                    <= self.BUDGET, name='budget')
        
        self._add_lower_level_constraints()

        # Set non-convex objective
        self.model.Params.NonConvex = 2
        self.model.update()


    def run(self):
        self.model.optimize()
        self._save_data()
    
    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save investment values
        self.data.investment_values = {i : {n : self.variables.P_investment[i][n].x for n in self.node_I[i]} for i in self.INVESTMENTS}
        
        # Save generation dispatch values
        self.data.generation_dispatch_values = {i : {n : {t : self.variables.p_g[i][n][t].x for t in self.TIMES} for n in self.node_I[i]} for i in self.INVESTMENTS}
        
        # Save uniform prices lambda
        self.data.lambda_ = {n : {t : self.variables.lmd[n][t].x for t in self.TIMES} for n in self.NODES}

    def display_results(self):
        print('Maximal NPV: \t{0} M€\n'.format(round(self.data.objective_value,2)))
        print('Investment Capacities:')
        for tech, investment in self.data.investment_values.items():
            capex = 0
            for node, value in investment.items():
                capex += value*self.CAPEX[tech]
                if value > 0:
                    print(f"{tech} at {node}: \t{round(value,2)} MW")
            print(f"Capital cost for {tech}: \t\t{round(capex,2)} M€\n")

    def plotNetwork(self):
        # create empty net
        net = pp.create_empty_network()
        cwd = os.path.dirname(__file__)
        bus_map = pd.read_csv(cwd + '/data/bus_map.csv', delimiter=';')
        
        line_map = pd.read_csv(cwd + '/data/lines.csv', delimiter=';')
        
        # Create buses
        for i in range(len(bus_map)):
            pp.create_bus(net, vn_kv=0.4, index = i,
                        geodata=(bus_map['x-coord'][i], -bus_map['y-coord'][i]),
                        name=bus_map['Bus'][i])
            
            for j in range(len(self.map_g[bus_map['Bus'][i]])):
                pp.create_gen(net, bus=i, p_mw=100)
            for j in range(len(self.map_d[bus_map['Bus'][i]])):
                pp.create_load(net, bus=i, p_mw=100)
            for j in range(len(self.map_w[bus_map['Bus'][i]])):
                pp.create_sgen(net, bus=i, p_mw=100)#, vm_pu=1.05)
            for g_type, nodal_investments in self.data.investment_values.items():
                for node, investment_size in nodal_investments.items():
                    if investment_size > 0 and node == bus_map['Bus'][i]:
                        if g_type == "Gas":
                            pp.create_gen(net, bus=i, p_mw=investment_size)
                            #net.gen.at[gen_idx, 'color'] = 'red'
                            net.bus.at[i, 'color'] = 'darkred'
                        elif g_type=="Nuclear":
                            pp.create_gen(net, bus=i, p_mw=investment_size)
                            net.bus.at[i, 'color'] = 'darkviolet'
                        elif g_type == "Solar":
                            pp.create_sgen(net, bus=i, p_mw=investment_size)
                            net.bus.at[i, 'color'] = 'darkorange'
                        elif g_type == "Offshore Wind":
                            pp.create_gen(net, bus=i, p_mw=investment_size)
                            net.bus.at[i, 'color'] = 'darkblue'
                        else:
                            pp.create_gen(net, bus=i, p_mw=investment_size)
                            net.bus.at[i, 'color'] = 'lightblue'
            

        # Create lines
        for i in range(len(line_map)):
            pp.create_line_from_parameters(net,
                    from_bus=    int(line_map['FromBus'][i][1:])-1,
                    to_bus=     int(line_map['ToBus'][i][1:])-1,
                    length_km=2,
                    name='L'+str(i+1),
                    r_ohm_per_km=0.2,
                    x_ohm_per_km=0.07,
                    c_nf_per_km=0.3,
                    max_i_ka=100)
        
        # plot the network
        size = 5
        d_c = plot.create_load_collection(net, loads=net.load.index, size=size)
        
        
        #gen_colors = net.gen['color'].fillna('black').tolist()  # Default to black if no color set
        gen_c = plot.create_gen_collection(net, gens=net.gen.index, size=size, orientation=0)
        wind_c = plot.create_sgen_collection(net, sgens=net.sgen.index, size=size, orientation=3.14/2)
        
        bus_colors = net.bus['color'].fillna('black').tolist()  # Default to blue if no color set
        bc = plot.create_bus_collection(net, buses=net.bus.index, size=size, 
                                        zorder=10, color=bus_colors)
        
        lc = plot.create_line_collection(net, lines=net.line.index, zorder=1, use_bus_geodata=True, color='grey')
        
        # Make a legend for investments
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, label='Gas'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkviolet', markersize=10, label='Nuclear'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=10, label='Solar'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=10, label='Offshore Wind'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Onshore Wind'),
            ]


        plot.draw_collections([lc, d_c, gen_c, wind_c, bc])
        plt.title("Network", fontsize=30)
        plt.legend(handles=legend_elements, loc='upper right')
        plt.show()



if __name__ == '__main__':
    # Initialize investment planning model
    ip = InvestmentPlanning(hours=24, budget=450, timelimit=200, carbontax=50, seed=38)
    # Build model
    ip.build_model()
    # Run optimization
    ip.run()
    # Display results
    ip.display_results()
    ip.plotNetwork()
#%%