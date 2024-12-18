#%%
import gurobipy as gb
from network import Network
from gurobipy import GRB
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt
import os
import time
from matplotlib.lines import Line2D
from iterative_ip import NodalClearing, NodalIP
from common_methods import CommonMethods


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class BilevelNodalIP(Network, CommonMethods):
    
    def __init__(self, 
                 chosen_hours:list[str] = ['T1', 'T2', 'T3', 'T4', 'T5',],
                 budget:float = 100,
                 timelimit:float = 100,
                 carbontax:float = 50,
                 seed:int = 42,
                 lmd_ub:dict[dict[float]] = None,
                 lmd_lb:dict[dict[float]] = None,
                 min_npv:float = 0,
                 max_npv:float = GRB.INFINITY,
                 bounded:bool = False,
                 invest_bound:float = GRB.INFINITY,
                 investments_guess:dict[str, dict[str, float]] = None,
                ):
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        self.data.objective_value = None # set objective value to None until optimization is performed
        
        self.chosen_hours = chosen_hours # set chosen hours for optimization
        self.T = len(chosen_hours) # set number of hours in optimization
        self.carbontax = carbontax # set carbon tax in €/tCO2
        self.BUDGET = budget # set budget for capital costs in M€
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)
        self.root_node = 'N1' # set root node for reference angle constraint
        self.bounded = bounded # set bounded flag for investment planning model
        self.lmd_ub = lmd_ub # set upper bounds for lambda in the investment planning model (from nodal clearing)
        self.lmd_lb = lmd_lb # set lower bounds for lambda in the investment planning model (from nodal clearing with price-taker investments)
        self.min_npv = min_npv # set lower bound NPV value for investment planning model
        self.max_npv = max_npv # set upper bound NPV value for investment planning model
        self.invest_bound = invest_bound # set upper bound for investments in generation technologies

        self._initialize_fluxes_demands()
        self._initialize_costs()
        self._compute_bounds()

    def _compute_bounds(self):
        # Big M notation and maximum values:
        self.max_investment_capacity = 10*max(self.P_G_max[g] for g in self.GENERATORS) # Big M for binary variables
        self.max_dual_values = 10**3 # Big M for dual variables
        self.max_demand = max(self.P_D[t][d] for d in self.DEMANDS for t in self.TIMES) # Maximum demand for a single demand
        self.max_production = self.max_investment_capacity # Maximum production capacity
        self.max_offer = max(self.C_offer.values()) # Maximum offer price
        self.max_bid = max(self.U_D.values()) # Maximum bid price
        self.max_lambda = max(self.max_offer, self.max_bid) # Maximum lambda value
        self.min_lambda = min(min(self.C_offer.values()), min(self.U_D.values())) # Minimum lambda value
        # Maximum delta in theta if all power would flow on the smallest line:
        self.max_theta_delta = max(self.demand_dict.values())/min(self.L_susceptance.values())
        self.max_theta = self.max_theta_delta / 2 # theta can be both positive and negative

    def _add_lower_level_variables(self):
        # Define primal variables of lower level KKTs
        self.variables.p_g          = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d          = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta        = {n: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.NODES}
        self.variables.flow         = {l: {t:       self.model.addVar(lb=-self.L_cap[l],ub=self.L_cap[l],name='flow_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

        # Define dual variables of lower level KKTs
        self.variables.nu           = {t:           self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Dual for reference angle constraint at time {0}'.format(t)) for t in self.TIMES}
        self.variables.lmd          = {n: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='spot price at time {0} in node {1}'.format(t,n)) for t in self.TIMES} for n in self.NODES} # Hourly spot price (€/MWh)
        self.variables.xi           = {l: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Dual for line {0} at time {1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}
        self.variables.mu_under     = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='Dual for lb on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over      = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='Dual for ub on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under  = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.sigma_over   = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.rho_over     = {l: {t:       self.model.addVar(lb=0,             ub=GRB.INFINITY, name='Dual for ub on line {0} at time {1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}
        self.variables.rho_under    = {l: {t:       self.model.addVar(lb=0,             ub=GRB.INFINITY, name='Dual for lb on line {0} at time {1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

        # Add binary auxiliary variables for bi-linear constraints
        self.variables.b1           = {g: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b1_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.b2           = {g: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b2_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.b3           = {d: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b3_{0}_{1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.b4           = {d: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b4_{0}_{1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.b5           = {l: {t:       self.model.addVar(vtype=GRB.BINARY, name='b5_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}
        self.variables.b6           = {l: {t:       self.model.addVar(vtype=GRB.BINARY, name='b6_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

    def _add_lower_level_variables_bounded(self):
        # Define primal variables of lower level KKTs
        self.variables.p_g          = {g: {n: {t:   self.model.addVar(lb=0,                 ub=self.max_production,         name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d          = {d: {n: {t:   self.model.addVar(lb=0,                 ub=self.max_demand,             name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta        = {n: {t:       self.model.addVar(lb=-self.max_theta,   ub=self.max_theta,             name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.NODES}
        self.variables.flow         = {l: {t:       self.model.addVar(lb=-self.L_cap[l],    ub=self.L_cap[l],               name='flow_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

        # Define dual variables of lower level KKTs
        self.variables.nu           = {t:           self.model.addVar(lb=-self.max_theta,   ub=self.max_theta,              name='Dual for reference angle constraint at time {0}'.format(t)) for t in self.TIMES}
        self.variables.lmd          = {n: {t:       self.model.addVar(lb=(self.lmd_lb[n][t] if self.lmd_lb else self.min_lambda),
                                                                      ub=(self.lmd_ub[n][t] if self.lmd_ub else self.max_lambda),
                                                                      name='spot price at time {0} in node {1}'.format(t,n))
                                                                      for t in self.TIMES} for n in self.NODES} # Hourly spot price (€/MWh)
        self.variables.xi           = {l: {t:       self.model.addVar(lb=-2*self.max_lambda,ub=2*self.max_lambda,           name='Dual for line {0} at time {1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}
        self.variables.mu_under     = {g: {n: {t:   self.model.addVar(lb=0,                 ub=self.C_offer[g],             name='Dual for lb on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over      = {g: {n: {t:   self.model.addVar(lb=0,                 ub=(self.lmd_ub[n][t] if self.lmd_ub else self.max_offer),    name='Dual for ub on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under  = {d: {n: {t:   self.model.addVar(lb=0,                 ub=(self.lmd_ub[n][t] if self.lmd_ub else self.max_bid),    name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.sigma_over   = {d: {n: {t:   self.model.addVar(lb=0,                 ub=self.U_D[d],                 name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.rho_over     = {l: {t:       self.model.addVar(lb=0,                 ub=2*self.max_lambda,           name='Dual for ub on line {0} at time {1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}
        self.variables.rho_under    = {l: {t:       self.model.addVar(lb=0,                 ub=2*self.max_lambda,           name='Dual for lb on line {0} at time {1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

        # Add binary auxiliary variables for bi-linear constraints
        self.variables.b1           = {g: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b1_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.b2           = {g: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b2_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.b3           = {d: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b3_{0}_{1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.b4           = {d: {n: {t:   self.model.addVar(vtype=GRB.BINARY, name='b4_{0}_{1}'.format(d, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.b5           = {l: {t:       self.model.addVar(vtype=GRB.BINARY, name='b5_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}
        self.variables.b6           = {l: {t:       self.model.addVar(vtype=GRB.BINARY, name='b6_{0}_{1}'.format(l, t)) for t in self.TIMES} for l in self.LINES}

    def _add_primal_lower_level_constraints(self):
        """ Constraint ensuring value of flow is DC-OPF related to voltage angles """
        self.constraints.flow       = self.model.addConstrs((self.variables.flow[l][t] == self.L_susceptance[l] * (self.variables.theta[self.node_L_from[l]][t] - self.variables.theta[self.node_L_to[l]][t])
                                                            for l in self.LINES for t in self.TIMES), name = "flow")
        # Generation capacity limits:
        self.constraints.p_gen_cap  = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g]
                                                            for g in self.GENERATORS for t in self.TIMES for n in self.node_G[g]), name = "gen_cap_generators")
        self.constraints.p_wt_cap   = self.model.addConstrs((self.variables.p_g[w][n][t] <= self.P_W[t][w]
                                                            for w in self.WINDTURBINES for t in self.TIMES for n in self.node_W[w]), name = "gen_cap_windturbines")
        self.constraints.p_inv_cap  = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.variables.P_investment[i][n] * self.fluxes[i][t] * self.cf[i]
                                                            for i in self.INVESTMENTS for t in self.TIMES for n in self.node_I[i]), name = "gen_cap_investments")
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
        # Reference voltage angle:
        self.constraints.ref_angle  = self.model.addConstrs((self.variables.theta[self.root_node][t] == 0
                                                            for t in self.TIMES), name = "ref_angle")

    def _add_dual_lower_level_constraints(self):
        """ KKT for lagrange objective derived wrt. generation variables """
        self.constraints.gen_lag    = self.model.addConstrs((self.C_offer[g] - self.variables.lmd[n][t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0
                                                            for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES), name = "derived_lagrange_generators")
        
        """ KKT for lagrange objective derived wrt. demand variables """
        self.constraints.dem_lag    = self.model.addConstrs((-self.U_D[d] + self.variables.lmd[n][t] - self.variables.sigma_under[d][n][t] + self.variables.sigma_over[d][n][t] == 0
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "derived_lagrange_demand")
        
        """ KKT for lagrange objective derived wrt. line flow variables """
        self.constraints.line_f_lag = self.model.addConstrs(((self.variables.lmd[self.node_L_from[l]][t] - self.variables.lmd[self.node_L_to[l]][t] - self.variables.rho_under[l][t] + self.variables.rho_over[l][t] + self.variables.xi[l][t] == 0)
                                                            for l in self.LINES for t in self.TIMES), name = "derived_lagrange_line_from")
        
        """ KKT for lagrange objective derived wrt. nodal voltage angle variables """
        self.constraints.angle_lags = self.model.addConstrs((gb.quicksum(-self.variables.xi[l][t]*self.L_susceptance[l] for l in self.map_from[n])
                                                            - gb.quicksum(-self.variables.xi[l][t]*self.L_susceptance[l] for l in self.map_to[n])
                                                            + (self.variables.nu[t] if n == self.root_node else 0) == 0
                                                            for n in self.NODES for t in self.TIMES), name = "derived_lagrange_angles")

        """ KKT for generation minimal production. Bi-linear are replaced by linearized constraints. Lower bound. """
        # Constraint mu_under == 0      for production units if b1 == 1:
        # Constraint p_g == 0           for production units if b1 == 0:
        # Existing Generators:
        self.constraints.gen_l_1    = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.P_G_max[g]
                                                            for g in self.GENERATORS for n in self.node_G[g] for t in self.TIMES), name = "generator_under_1")
        # Existing Wind Turbines: 
        self.constraints.wt_l_1     = self.model.addConstrs((self.variables.p_g[w][n][t] <= self.variables.b1[w][n][t] * self.P_W[t][w]
                                                            for w in self.WINDTURBINES for n in self.node_W[w] for t in self.TIMES), name = "windturbine_under_1")
        # New Investments:
        self.constraints.inv_l_1    = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.variables.b1[i][n][t] * self.variables.P_investment[i][n] * self.fluxes[i][t] * self.cf[i]
                                                            for i in self.INVESTMENTS for n in self.node_I[i] for t in self.TIMES), name = "investment_under_1")
        # Constraint mu_under == 0 for production units if b1 == 1:
        self.constraints.prod_l_2   = self.model.addConstrs((self.variables.mu_under[g][n][t] <= self.C_offer[g] * (1 - self.variables.b1[g][n][t])
                                                            for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES), name = "production_under_2") 
        """ KKT for production capacities. Bi-linear are replaced by linearized constraints. Upper bound. """
        # Constraint mu_over == 0       for production units if b2 == 1:
        # Constraint p_g == p_g_max     for production units if b2 == 0:
        # Existing Generators:
        self.constraints.gen_u_1    = self.model.addConstrs((self.P_G_max[g] * (1 - self.variables.b2[g][n][t]) <= self.variables.p_g[g][n][t]
                                                            for g in self.GENERATORS for n in self.node_G[g] for t in self.TIMES), name = "gen_upper_generators_1")
        self.constraints.gen_u_2    = self.model.addConstrs((self.variables.mu_over[g][n][t] <= self.max_dual_values * (1 - self.variables.b2[g][n][t])
                                                            for g in self.GENERATORS for n in self.node_G[g] for t in self.TIMES), name = "gen_upper_generators_2")
        # Existing Wind Turbines:
        self.constraints.wt_u_1     = self.model.addConstrs((self.P_W[t][w] * (1 - self.variables.b2[w][n][t]) <= self.variables.p_g[w][n][t]
                                                            for w in self.WINDTURBINES for n in self.node_W[w] for t in self.TIMES), name = "gen_upper_windturbines_1")
        self.constraints.wt_u_2     = self.model.addConstrs((self.variables.mu_over[w][n][t] <= self.max_dual_values * (1 - self.variables.b2[w][n][t])
                                                            for w in self.WINDTURBINES for n in self.node_W[w] for t in self.TIMES), name = "gen_upper_windturbines_2")
        # New Investments:
        self.constraints.inv_u_1    = self.model.addConstrs((self.variables.P_investment[i][n] * self.fluxes[i][t] * self.cf[i] - self.max_investment_capacity * self.variables.b2[i][n][t] <= self.variables.p_g[i][n][t]
                                                            for i in self.INVESTMENTS for n in self.node_I[i] for t in self.TIMES), name = "gen_upper_investments_1")
        self.constraints.inv_u_2    = self.model.addConstrs((self.variables.mu_over[i][n][t] <= self.max_dual_values * (1 - self.variables.b2[i][n][t])
                                                            for i in self.INVESTMENTS for n in self.node_I[i] for t in self.TIMES), name = "gen_upper_investments_2")
        
        """ KKT for demand capacities. Bi-linear are replaced by linearized constraints. Lower bound. """
        # Constraint sigma_under == 0   for demands if b3 == 1:
        # Constraint p_d == 0           for demands if b3 == 0:
        self.constraints.dem_l_1    = self.model.addConstrs((self.variables.p_d[d][n][t] <= self.variables.b3[d][n][t] * self.P_D[t][d]
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_under_1")
        self.constraints.dem_l_2    = self.model.addConstrs((self.variables.sigma_under[d][n][t] <= self.max_dual_values * (1 - self.variables.b3[d][n][t])
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_under_2")
        """ KKT for demand capacities. Bi-linear are replaced by linearized constraints. Upper bound. """
        # Constraint sigma_over == 0    for demands if b4 == 1:
        # Constraint p_d == P_D         for demands if b4 == 0:
        self.constraints.dem_u_1    = self.model.addConstrs((self.P_D[t][d] * (1 - self.variables.b4[d][n][t]) <= self.variables.p_d[d][n][t]
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_upper_1")
        self.constraints.dem_u_2    = self.model.addConstrs((self.variables.sigma_over[d][n][t] <= self.U_D[d] * (1 - self.variables.b4[d][n][t])
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_upper_2")
        
        """ KKT for line flow constraints. Bi-linear are replaced by linearized constraints. Lower bound. """
        # Constraint rho_under == 0     for lines if b5 == 1:
        # Constraint flow == -L_cap     for lines if b5 == 0:
        self.constraints.line_l_1   = self.model.addConstrs((self.variables.flow[l][t] <= -self.L_cap[l] + self.variables.b5[l][t] * 2 * self.L_cap[l]
                                                            for l in self.LINES for t in self.TIMES), name = "line_lower_1")
        self.constraints.line_l_2   = self.model.addConstrs((self.variables.rho_under[l][t] <= self.max_dual_values * (1 - self.variables.b5[l][t])
                                                            for l in self.LINES for t in self.TIMES), name = "line_lower_2")
        """ KKT for line flow constraints. Bi-linear are replaced by linearized constraints. Upper bound. """
        # Constraint rho_over == 0      for lines if b6 == 1:
        # Constraint flow == L_cap      for lines if b6 == 0:
        self.constraints.line_u_1   = self.model.addConstrs((self.variables.flow[l][t] >= self.L_cap[l] - self.variables.b6[l][t] * 2 * self.L_cap[l]
                                                            for l in self.LINES for t in self.TIMES), name = "line_upper_1")
        self.constraints.line_u_2   = self.model.addConstrs((self.variables.rho_over[l][t] <= self.max_dual_values * (1 - self.variables.b6[l][t])
                                                            for l in self.LINES for t in self.TIMES), name = "line_upper_2")
        
    def _add_lower_level_constraints(self):
        self._add_primal_lower_level_constraints()
        self._add_dual_lower_level_constraints()        

    def build_model(self):
        self.model = gb.Model(name='Investment Planning')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        # Investment in generation technologies (in MW)
        self.variables.P_investment = {i : {n : self.model.addVar(lb=0, ub=min(self.BUDGET/self.CAPEX[i], self.invest_bound),
                                                                  name='investment in tech {0} at node {1}'.format(i, n)) for n in self.node_I[i]} for i in self.INVESTMENTS}
        self.variables.npv = self.model.addVar(lb=self.min_npv, ub=self.max_npv, name='NPV')

        # Get lower level variables
        if self.bounded:
            self._add_lower_level_variables_bounded()
        else:
            self._add_lower_level_variables()

        self.model.update()

        """ Initialize objective to maximize NPV [M€] """
        # Define costs (annualized capital costs + fixed and variable operational costs)
        self.costs = gb.quicksum( self.variables.P_investment[i][n] * (self.AF[i] * self.CAPEX[i] + self.f_OPEX[i])
                            + 8760/self.T * gb.quicksum(self.v_OPEX[i] * self.variables.p_g[i][n][t] for t in self.TIMES)
                            for i in self.INVESTMENTS for n in self.node_I[i])
        # Define revenue (sum of generation revenues) [M€]
        self.revenue = (8760 / self.T / 10**6) * gb.quicksum(self.variables.lmd[n][t] * self.variables.p_g[i][n][t]
                                            for i in self.INVESTMENTS for t in self.TIMES for n in self.node_I[i])
        
        # Define NPV
        self.model.addConstr(self.variables.npv == self.revenue - self.costs)

        # Set objective to maximize NPV
        self.model.setObjective(self.variables.npv, gb.GRB.MAXIMIZE)

        self.model.update()

        """ Initialize constraints """
        # Budget constraints
        self.constraints.budget = self.model.addConstr(gb.quicksum(
                                    self.variables.P_investment[i][n] * self.CAPEX[i] for i in self.INVESTMENTS for n in self.node_I[i])
                                    <= self.BUDGET, name='budget')
        
        self._add_lower_level_constraints()

        self.model.Params.NonConvex = 2 # Set non-convex objective
        #self.model.Params.MIPFocus = 1 # Set focus on MIP in terms of balanced (0), feasible (1), optimal (2), or bound (3) solutions
        self.model.Params.MIPGap = 0.01 # Set MIP gap to 1%
        self.model.update()

    def _calculate_capture_prices(self):
        # Calculate capture price
        self.data.capture_prices = { i:
                                    {n:
                                        round((sum(self.data.lambda_[n][t] * self.data.investment_dispatch_values[t][i][n] for t in self.TIMES) /
                                        sum(self.data.investment_dispatch_values[t][i][n] for t in self.TIMES)), 3)
                                        if self.data.investment_values[i][n] > 0
                                        else
                                        None
                                    for n in self.node_I[i]} 
                                    for i in self.INVESTMENTS}

    def _save_data(self):
        solution = False # Whether a solution was found
        if self.model.status == GRB.Status.OPTIMAL:
            self.data.status = 'Optimal'
            solution = True
        # Check if a solution was found
        elif self.model.status == GRB.Status.INFEASIBLE:
            self.data.status = 'Infeasible'
        elif self.model.status == GRB.Status.TIME_LIMIT:
            self.data.status = 'Time limit reached'
            if self.model.SolCount > 0:
                solution = True
        elif self.model.status == GRB.Status.UNBOUNDED:
            self.data.status = 'Unbounded'
        else:
            self.data.status = 'Other'
            if self.model.SolCount > 0:
                solution = True

        if solution:
            # Save objective value
            self.data.objective_value = self.model.ObjVal
            
            # Save investment values
            self.data.investment_values = {i : {n : round(self.variables.P_investment[i][n].x, 3) for n in self.node_I[i]} for i in self.INVESTMENTS}
            self.data.capacities = {t :
                                    {**{i : {n: round(self.data.investment_values[i][n]*self.fluxes[i][t]*self.cf[i], 3) for n in self.node_I[i]} for i in self.INVESTMENTS},
                                    **{g : {n: self.P_G_max[g] for n in self.node_G[g]} for g in self.GENERATORS},
                                    **{w : {n: self.P_W[t][w] for n in self.node_W[w]} for w in self.WINDTURBINES}}
                                    for t in self.TIMES}
            
            # Save generation dispatch values
            self.data.investment_dispatch_values = {t : {i : {n : round(self.variables.p_g[i][n][t].x, 3) for n in self.node_I[i]} for i in self.INVESTMENTS} for t in self.TIMES}
            self.data.generator_dispatch_values = {t : {g : {n : round(self.variables.p_g[g][n][t].x, 3) for n in self.node_G[g]} for g in self.GENERATORS} for t in self.TIMES}
            self.data.windturbine_dispatch_values = {t : {w : {n : round(self.variables.p_g[w][n][t].x, 3) for n in self.node_W[w]} for w in self.WINDTURBINES} for t in self.TIMES}
            self.data.all_dispatch_values = {t : {g : {n : round(self.variables.p_g[g][n][t].x, 3) for n in self.node_production[g]} for g in self.PRODUCTION_UNITS} for t in self.TIMES}
            
            # Save demand dispatch values
            self.data.demand_dispatch_values = {t : {d: {n : round(self.variables.p_d[d][n][t].x, 3) for n in self.node_D[d]} for d in self.DEMANDS} for t in self.TIMES}

            # Save uniform prices lambda
            self.data.lambda_ = {n : {t : round(self.variables.lmd[n][t].x, 3) for t in self.TIMES} for n in self.NODES}

            # Save voltage angles
            self.data.theta = {n : {t : round(self.variables.theta[n][t].x, 3) for t in self.TIMES} for n in self.NODES}

            self.data.xi = {l : {t: round(self.variables.xi[l][t].x, 3) for t in self.TIMES} for l in self.LINES}

            self.data.rho_over = {l : {t: round(self.variables.rho_over[l][t].x, 3) for t in self.TIMES} for l in self.LINES}
            self.data.rho_under = {l : {t: round(self.variables.rho_under[l][t].x, 3) for t in self.TIMES} for l in self.LINES}

            self.data.flow = {l : {t: round(self.variables.flow[l][t].x, 3) for t in self.TIMES} for l in self.LINES}

            self._calculate_capture_prices()
        else:
            print('No solution found.')
            print('Status: {0}'.format(self.data.status))

    def run(self):
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        self._save_data()
    
    def display_results(self):
        if self.data.objective_value:
            print('Maximal NPV: \t{0} M€\n'.format(round(self.data.objective_value,1)))
            print('Investment Capacities:')
            for g_type, nodal_investments in self.data.investment_values.items():
                capex = 0
                for node, investment_size in nodal_investments.items():
                    capex += investment_size*self.CAPEX[g_type]
                    if investment_size > 0:
                        print(f"{g_type} at {node}: \t{round(investment_size,1)} MW")
                if capex > 0: 
                    print(f"Capital cost for {g_type}: \t\t{round(capex,1)} M€\n")
        else:
            print('No solution found.')
    
    def get_objective_value(self):
        return self.data.objective_value

    def plot_network(self):
        # create empty net
        net = pp.create_empty_network()
        cwd = os.path.dirname(__file__)
        bus_map = pd.read_csv(cwd + '/data/bus_map.csv', delimiter=';') # Move to Network class?
        
        line_map = pd.read_csv(cwd + '/data/lines.csv', delimiter=';')
        
        # Create buses
        for i in range(len(bus_map)):
            pp.create_bus(net, vn_kv=0.4, index = i,
                        geodata=(bus_map['x-coord'][i], -bus_map['y-coord'][i]),
                        name=bus_map['Bus'][i])
            
            for _ in range(len(self.map_g[bus_map['Bus'][i]])):
                pp.create_gen(net, bus=i, p_mw=100)
            for _ in range(len(self.map_d[bus_map['Bus'][i]])):
                pp.create_load(net, bus=i, p_mw=100)
            for _ in range(len(self.map_w[bus_map['Bus'][i]])):
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

    def plot_prices(self):     
        # Define a list of colors and line styles
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        linestyles = ['-', '--']

        # Plot the nodal time series as stairs plots with unique colors and line styles
        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        # Plot the first 12 graphs in the upper plot
        for i, node in enumerate(self.NODES[:12]):
            lambda_values = [self.data.lambda_[node][t] for t in self.TIMES]
            color = colors[i % len(colors)]
            linestyle = linestyles[i//6 % len(linestyles)]
            ax1.plot(self.TIMES, lambda_values, drawstyle='steps', label=node, color=color, linestyle=linestyle)

        # Plot the remaining graphs in the lower plot
        for i, node in enumerate(self.NODES[12:]):
            lambda_values = [self.data.lambda_[node][t] for t in self.TIMES]
            color = colors[i % len(colors)]
            linestyle = linestyles[i//6 % len(linestyles)]
            ax2.plot(self.TIMES, lambda_values, drawstyle='steps', label=node, color=color, linestyle=linestyle)

        # Add labels and legend
        ax1.set_ylabel('Price [$/MWh]', fontsize=16)
        ax2.set_xlabel('Time', fontsize=16)
        ax2.set_ylabel('Price [$/MWh]', fontsize=16)
        ax1.legend(loc = 'upper left', fontsize=12)
        ax2.legend(loc = 'upper left', fontsize=12)

        # Show the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        plt.show()

    def _get_demand_curve_info(self, T: str):
        bids = pd.Series(self.U_D).sort_values(ascending=False)
        #dispatch = pd.Series(self.data.demand_dispatch_values[T]).loc[bids.index]
        demand = pd.Series(self.P_D[T]).loc[bids.index]
        #dispatch_cumulative = dispatch.copy()
        demand_cumulative = demand.cumsum()
        #dispatch_cumulative = pd.concat([pd.Series([0]), dispatch_cumulative])
        demand_cumulative = pd.concat([pd.Series([0]), demand_cumulative])
        bids = pd.concat([pd.Series([bids.iloc[0]]), bids])
        
        return bids, demand_cumulative, #dispatch_cumulative
    
    def _get_supply_curve_info(self, T: str):
        self.C_offer_modified = self.C_offer.copy()
        for i in self.INVESTMENTS:
            del self.C_offer_modified[i]
            for n in self.node_I[i]:
                self.C_offer_modified[(i,n)] = self.C_offer[i]
        offers = pd.Series(self.C_offer_modified).sort_values(ascending=True)
        #dispatch = pd.Series(self.data.all_dispatch_values[T]).loc[offers.index]
        capacity = pd.Series()
        for gen, offer in offers.items():
            if gen in self.GENERATORS:
                capacity[gen] = self.P_G_max[gen]
            elif gen in self.WINDTURBINES:
                capacity[gen] = self.P_W[T][gen]
            else:
                capacity[str(gen)] = self.data.investment_values[gen[0]][gen[1]]*self.fluxes[gen[0]][T]*self.cf[gen[0]]
        #capacity = pd.Series(self.data.capacities[T]).loc[offers.index]
        #dispatch_cumulative = dispatch.cumsum()
        capacity_cumulative = capacity.cumsum()
        #dispatch_cumulative = pd.concat([pd.Series([0]), dispatch_cumulative])
        capacity_cumulative = pd.concat([pd.Series([0]), capacity_cumulative])
        offers = pd.concat([pd.Series(offers.iloc[0]), offers])
        
        return offers, capacity_cumulative, #dispatch_cumulative
        
    def plot_supply_demand_curve(self, T: str):

        offers, capacity_cumulative = self._get_supply_curve_info(T)
        bids, demand_cumulative = self._get_demand_curve_info(T)
        plt.step(np.array(capacity_cumulative), np.array(offers), label='Supply', )
        plt.step(np.array(demand_cumulative), np.array(bids), label='Demand', )
        
        plt.xlabel('Quantity [MWh]')
        plt.ylabel('Power Price [€/MWh]')
        plt.title('Supply/Demand curve for time {0}'.format(T))
        plt.xlim(0, max(capacity_cumulative.max(), demand_cumulative.max())*1.1)
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    # Initialize investment planning model
    seed = 38
    timelimit = 3600*10
    budget = 10000
    carbontax = 60

    # Intended initialization of chosen_hours:
    chosen_days = [190] # if it should be random then do: chosen_days = None
    chosen_days = range(365)
    chosen_hours = list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24))
    
    ## Initialization of small model for testing:
    runtimes = {}
    gb_runtimes = {}
    objective_values = {}
    model_options = {'bounds': 1,
                'price_ub': 2,
                'max_npv': 3,
                'min_npv': 4,}
    
    for n_hours in [5, 24]:
        runtimes[str(n_hours)] = {}
        gb_runtimes[str(n_hours)] = {}
        objective_values[str(n_hours)] = {}
        for scen in [5]:
            first_hour = 19
            chosen_hours = list('T{0}'.format(i) for i in range(first_hour, first_hour+n_hours))

            t_start = time.time()

            bounded = model_options['bounds'] <= scen
            price_ub = None
            price_lb = None
            min_npv = 0
            max_npv = GRB.INFINITY

            if model_options['price_ub'] <= scen: # Whether we include upper bounds for lambda in the investment planning model
                nc_org = NodalClearing(timelimit=timelimit, carbontax=carbontax, seed=seed, chosen_hours=chosen_hours)
                nc_org.build_model()
                nc_org.run()
                price_ub        = nc_org.data.lambda_
                if model_options['max_npv'] <= scen: # Whether we include an upper bound for NPV in the investment planning model
                    n_ip = NodalIP(chosen_hours=chosen_hours, budget=budget, timelimit=timelimit, carbontax=carbontax, seed=seed, lmd=price_ub)
                    n_ip.build_model()
                    n_ip.run()
                    investments     = n_ip.data.investment_values
                    max_npv         = n_ip.data.objective_value
                    if model_options['min_npv'] <= scen: # Whether we include a lower bound for NPV in the investment planning model
                        nc_org_2 = NodalClearing(timelimit=timelimit, carbontax=carbontax, seed=seed, chosen_hours=chosen_hours, P_investment=investments)
                        nc_org_2.build_model()
                        nc_org_2.run()
                        min_npv         = nc_org_2.data.npv
                        #price_lb        = nc_org_2.data.lambda_ # Does not work as intended

            ip = BilevelNodalIP(chosen_hours=chosen_hours,
                                budget=budget,
                                timelimit=timelimit,
                                carbontax=carbontax,
                                seed=seed,
                                lmd_ub=None,
                                lmd_lb=price_lb,
                                max_npv=max_npv,
                                min_npv=min_npv,
                                bounded=bounded,
                                )
            ip.build_model()
            ip.run()

            t_end = time.time()
            runtime = t_end - t_start
            runtimes[str(n_hours)][str(scen)] = round(runtime,2)
            gb_runtimes[str(n_hours)][str(scen)] = round(ip.model.Runtime,2)
            objective_values[str(n_hours)][str(scen)] = ip.get_objective_value()

            # Display results
            ip.display_results()
            print('Runtime: {0} seconds'.format(round(runtime,2)))
    print(runtimes)
    print(gb_runtimes)
    print(objective_values)
    print()
    
    #ip.plot_network()
    #ip.plot_supply_demand_curve('T1')
    #ip.plot_prices()
    # Save investment planning data in a dataframe and then in a csv file
    investment_values_df = pd.DataFrame(ip.data.investment_values)
    investment_values_df.to_csv('results/investment_values,hours={n_hours},budget={budget},carbontax={carbontax},runtime={runtime}.csv'.format(n_hours=n_hours,budget=budget,carbontax=carbontax,runtime=ip.model.Runtime))
