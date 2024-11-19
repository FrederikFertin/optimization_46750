#%%
import gurobipy as gb
from network import Network
from gurobipy import GRB
import numpy as np
import random

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class InvestmentPlanning(Network):
    
    def __init__(self, hours:int = 24, budget:float = 100, timelimit:float=100): # initialize class
        super().__init__()
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
        #self.AF["Nuclear"] = 0.02

        if hours >= 24:
            assert hours % 24 == 0, "Hours must be a multiple of 24"
            days = hours // 24
            chosen_days = np.random.choice(range(365), days, replace=False)
            self.cf = {i: 1 for i in self.INVESTMENTS} # If typical days are used, set capacity factors to 1.

            self.offshore_flux = np.concatenate([self.offshore_hourly_2019[d*24: (d+1)*24].values for d in chosen_days])
            self.solar_flux = np.concatenate([self.solar_hourly_2019[d*24: (d+1)*24].values for d in chosen_days])
            self.onshore_flux = np.concatenate([self.onshore_hourly_2019[d*24: (d+1)*24].values for d in chosen_days])
            self.demand_profiles = np.concatenate([self.demand_hourly.iloc[d*24: (d+1)*24]['Demand'].values for d in chosen_days])
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
        self.BUDGET = budget # set budget for capital costs in M€
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)

        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS

    def _add_lower_level_variables(self):
        # Define variables of lower level KKTs
        self.variables.p_g   = {g: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES} for n in self.NODES} for g in self.PRODUCTION_UNITS}
        self.variables.p_d   = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.lmd   = {n: {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='spot price at time {0} in node {1}'.format(t,n)) for t in self.TIMES} for n in self.NODES} # Hourly spot price (€/MWh)
        self.variables.theta = {n: {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.NODES}

        
        self.variables.mu_under    = {g: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.NODES} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over     = {g: {n: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at node {1} at time {2}'.format(g, n, t)) for t in self.TIMES} for n in self.NODES} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.sigma_over  = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.rho_under   = {n: {m: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb between line {0} and line {1} at time {2}'.format(n, m, t)) for t in self.TIMES} for m in self.NODES} for n in self.NODES}
        self.variables.rho_over    = {n: {m: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub between line {0} and line {1} at time {2}'.format(n, m, t)) for t in self.TIMES} for m in self.NODES} for n in self.NODES}

        # Add binary auxiliary variables for bi-linear constraints
        self.variables.b1 = {g: {n: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b1_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.NODES} for g in self.PRODUCTION_UNITS}
        self.variables.b2 = {g: {n: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b2_{0}_{1}_{2}'.format(g, n, t)) for t in self.TIMES} for n in self.NODES} for g in self.PRODUCTION_UNITS}
        self.variables.b3 = {d: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b3_{0}_{1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.b4 = {d: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b4_{0}_{1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.b5 = {n: {m: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b5_{0}_{1}_{2}'.format(n, m, t)) for t in self.TIMES} for m in self.NODES} for n in self.NODES}
        self.variables.b6 = {n: {m: {t: self.model.addVar(vtype=gb.GRB.BINARY, name='b6_{0}_{1}_{2}'.format(n, m, t)) for t in self.TIMES} for m in self.NODES} for n in self.NODES}


    def _add_lower_level_constraints(self):
        M = max(self.P_G_max[g] for g in self.GENERATORS) # Big M for binary variables
        
        # KKT for lagrange objective derived wrt. generation variables
        self.constraints.gen_lagrange_generators   = self.model.addConstrs((self.C_G_offer[g] - self.variables.lmd[n][t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "derived_lagrange_generators")
        self.constraints.gen_lagrange_windturbines = self.model.addConstrs((self.v_OPEX['Offshore Wind'] - self.variables.lmd[t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "derived_lagrange_windturbines")
        self.constraints.gen_lagrange_investments  = self.model.addConstrs((self.v_OPEX[g] - self.variables.lmd[n][t] - self.variables.mu_under[g][n][t] + self.variables.mu_over[g][n][t] == 0 for g in self.INVESTMENTS for n in self.NODES for t in self.TIMES), name = "derived_lagrange_investments")
        
        # KKT for lagrange objective derived wrt. demand variables
        self.constraints.dem_lagrange = self.model.addConstrs((-self.U_D[d] + self.variables.lmd[n][t] - self.variables.sigma_under[d][t] + self.variables.sigma_over[d][t] == 0 for n in self.NODES for d in self.map_d[n] for t in self.TIMES), name = "derived_lagrange_demand")
        
        # KKT for lagrange objective derived wrt. line flow variables
        self.constraints.line_lagrange = self.model.addConstrs((self.L_susceptance[l] * (self.variables.lmd[n][t] - self.variables.rho_under[n][m][t] + self.variables.rho_over[n][m][t]) == 0 for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "derived_lagrange_line")

        # KKT for generation minimal production. Bi-linear are replaced by linearized constraints
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.P_G_max[g] for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.P_W[g] for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.b1[g][n][t] * self.variables.P_investment[g][n] for g in self.INVESTMENTS for n in self.NODES for t in self.TIMES), name = "gen_under_1")
        self.constraints.gen_under_2 = self.model.addConstrs((self.variables.mu_under[g][n][t] <= M * (1 - self.variables.b1[g][n][t]) for g in self.PRODUCTION_UNITS for n in self.NODES for t in self.TIMES), name = "gen_under_2") 

        # KKT for generation capacities. Bi-linear are replaced by linearized constraints
        self.constraints.gen_upper_generators_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g] + M * self.variables.b2[g][n][t] for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "gen_upper_generators_1")
        self.constraints.gen_upper_generators_2 = self.model.addConstrs((self.P_G_max[g] - M * self.variables.b2[g][n][t] <= self.variables.p_g[g][n][t] for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "gen_upper_generators_2")
        self.constraints.gen_upper_generators_3 = self.model.addConstrs((self.variables.mu_over[g][n][t] <= M * (1 - self.variables.b2[g][n][t]) for g in self.GENERATORS for n in self.NODES for t in self.TIMES), name = "gen_upper_generators_3")

        self.constraints.gen_upper_windturbines_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_W[t][g] + M * self.variables.b2[g][n][t] for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "gen_upper_windturbines_1")
        self.constraints.gen_upper_windturbines_2 = self.model.addConstrs((self.P_W[t][g] - M * self.variables.b2[g][n][t] <= self.variables.p_g[g][n][t] for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "gen_upper_windturbines_2")
        self.constraints.gen_upper_windturbines_3 = self.model.addConstrs((self.variables.mu_over[g][n][t] <= M * (1 - self.variables.b2[g][n][t]) for g in self.WINDTURBINES for n in self.NODES for t in self.TIMES), name = "gen_upper_windturbines_3")

        self.constraints.gen_upper_investments_1 = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.variables.P_investment[g][n] * self.fluxes[g][t_ix] + M * self.variables.b2[g][n][t] for g in self.INVESTMENTS for n in self.NODES for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments_1")
        self.constraints.gen_upper_investments_2 = self.model.addConstrs((self.variables.P_investment[g][n] * self.fluxes[g][t_ix] - M * self.variables.b2[g][n][t] <= self.variables.p_g[g][n][t] for g in self.INVESTMENTS for n in self.NODES for t_ix, t in enumerate(self.TIMES)), name = "gen_upper_investments_2")
        self.constraints.gen_upper_investments_3 = self.model.addConstrs((self.variables.mu_over[g][n][t] <= M * (1 - self.variables.b2[g][n][t]) for g in self.INVESTMENTS for n in self.NODES for t in self.TIMES), name = "gen_upper_investments_3")

        # KKT for demand constraints. Bi-linear are replaced by linearized constraints
        # self.constraints.dem_under = self.model.addConstrs((-self.variables.p_d[d][t] * self.variables.sigma_under[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_under_1 = self.model.addConstrs((self.variables.p_d[d][t] <= self.variables.b3[d][t] * M for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_under_2 = self.model.addConstrs((self.variables.sigma_under[d][t] <= M * (1 - self.variables.b3[d][t]) * M for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        
        # self.constraints.dem_upper_1 = self.model.addConstrs((self.variables.p_d[d][t] <= self.P_D[t][d] + M * self.variables.x[d][t] for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_1")
        self.constraints.dem_upper_2 = self.model.addConstrs((self.P_D[t][d] - M * self.variables.b4[d][t] <= self.variables.p_d[d][t] for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_3")
        self.constraints.dem_upper_3 = self.model.addConstrs((self.variables.sigma_over[d][t] <= M * (1 - self.variables.b4[d][t]) for d in self.DEMANDS for t in self.TIMES), name = "dem_upper_2")
        
        # KKT for line flow constraints. Bi-linear are replaced by linearized constraints
        self.constraints.line_under_1 = self.model.addConstrs((self.variables.rho_under[n][m][t] <= M * (1 - self.variables.b5[n][m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_under_1")
        self.constraints.line_under_2 = self.model.addConstrs((self.L_cap[l]/self.L_susceptance[l] - (1-self.variables.b5[n][m][t]) * M <= self.variables.theta[n][t] - self.variables.theta[m][t] for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_under_2")
        self.constraints.line_under_3 = self.model.addConstrs((self.variables.theta[n][t] - self.variables.theta[m][t] <= self.L_cap[l]/self.L_susceptance[l] + (1-self.variables.b5[n][m][t]) * M for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_under_3")
        
        self.constraints.line_over_1 = self.model.addConstrs((self.variables.rho_over[n][m][t] <= M * (1 - self.variables.b6[n][m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "line_over_1")
        self.constraints.line_over_2 = self.model.addConstrs((-self.L_cap[l]/self.L_susceptance[l] - (1-self.variables.b6[n][m][t]) * M <= self.variables.theta[n][t] - self.variables.theta[m][t] for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_over_2")
        self.constraints.line_over_3 = self.model.addConstrs((self.variables.theta[n][t] - self.variables.theta[m][t] <= -self.L_cap[l]/self.L_susceptance[l] + (1-self.variables.b6[n][m][t]) * M for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items()), name = "line_over_3")
    

        ### Primal constraints ###
        # Generation capacity limits
        self.constraints.gen_cap_generators   = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g] for g in self.GENERATORS for t in self.TIMES for n in self.NODES), name = "gen_cap_generators")
        self.constraints.gen_cap_windturbines = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_W[t][g] for g in self.WINDTURBINES for t in self.TIMES for n in self.NODES), name = "gen_cap_windturbines")
        self.constraints.gen_cap_investments  = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.variables.P_investment[i][n] * self.fluxes[i][t_ix] for i in self.INVESTMENTS for t_ix, t in enumerate(self.TIMES) for n in self.NODES), name = "gen_cap_investments")
        
        # Demand magnitude constraints
        self.constraints.dem_mag = self.model.addConstrs((self.variables.p_d[d][t] <= self.P_D[t][d] for d in self.DEMANDS for t in self.TIMES), name = "dem_mag")
        
        # Balancing constraint
        self.constraints.balance = self.model.addConstrs((gb.quicksum(self.variables.p_g[g][n][t] for g in self.PRODUCTION_UNITS) - gb.quicksum(self.variables.p_d[d][t] for d in self.map_d[n])
                                                           + gb.quicksum(self.L_susceptance[l] * (self.variables.theta[n][t] - self.variables.theta[m][t]) for m, l in self.map_n[n].items() for n in self.NODES) == 0  for t in self.TIMES for n in self.NODES), name = "balance")

        # Line capacity constraints
        self.constraints.line_cap_lower = self.model.addConstrs((- self.L_cap[l] <= self.L_susceptance[l] * (self.variables.theta[n][t] - self.variables.theta[m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "line_cap_lower")
        self.constraints.line_cap_upper = self.model.addConstrs((self.L_cap[l] >= self.L_susceptance[l] * (self.variables.theta[n][t] - self.variables.theta[m][t]) for n in self.NODES for t in self.TIMES for m, l in self.map_n[n].items() ), name = "line_cap_upper")

        # Reference voltage angle
        self.constraints.ref_angle = self.model.addConstrs((self.variables.theta['N1'][t] == 0 for t in self.TIMES), name = "ref_angle")

    def build_model(self):
        self.model = gb.Model(name='Investment Planning')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        # Investment in generation technologies (in MW)
        self.variables.P_investment = {i : {n : self.model.addVar(lb=0, ub=GRB.INFINITY, name='investment in tech {0} at node {1}'.format(i, n)) for n in self.NODES} for i in self.INVESTMENTS}
        
        # Get lower level variables
        self._add_lower_level_variables()

        self.model.update()

        """ Initialize objective to maximize NPV [M€] """
        # Define costs (annualized capital costs + fixed and variable operational costs)
        costs = gb.quicksum( self.variables.P_investment[i][n] * (self.AF[i] * self.CAPEX[i] + self.f_OPEX[i])
                            + 8760/self.T * gb.quicksum(self.v_OPEX[i] * self.variables.p_g[i][n][t] for t in self.TIMES)
                            for i in self.INVESTMENTS for n in self.NODES)
        # Define revenue (sum of generation revenues) [M€]
        revenue = (8760 / self.T / 10**6) * gb.quicksum(self.cf[i] * 
                                            self.variables.lmd[n][t] * self.variables.p_g[i][n][t]
                                            for i in self.INVESTMENTS for t in self.TIMES for n in self.NODES)
        
        # Define NPV, including magic constant
        npv = revenue - costs

        # Set objective
        self.model.setObjective(npv, gb.GRB.MAXIMIZE)

        self.model.update()

        """ Initialize constraints """
        # Budget constraints
        self.constraints.budget = self.model.addConstr(gb.quicksum(
                                    self.variables.P_investment[i][n] * self.CAPEX[i] for i in self.INVESTMENTS for n in self.NODES)
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
        self.data.investment_values = {i : {n : self.variables.P_investment[i][n].x for n in self.NODES} for i in self.INVESTMENTS}
        
        # Save generation dispatch values
        self.data.generation_dispatch_values = {g : {n : {t : self.variables.p_g[g][n][t].x for t in self.TIMES} for n in self.NODES} for g in self.INVESTMENTS}
        
        # Save uniform prices lambda
        self.data.lambda_ = {n : {t : self.variables.lmd[n][t].x for t in self.TIMES} for n in self.NODES}

    def display_results(self):
        print('Maximal NPV: \t{0} M€\n'.format(round(self.data.objective_value,2)))
        print('Investment Capacities:')
        for key, value in self.data.investment_values.items():
            print(f"{key}: \t\t{round(value,2)} MW")
            print(f"Capital cost: \t\t{round(value*self.CAPEX[key],2)} M€\n")



if __name__ == '__main__':
    # Initialize investment planning model
    ip = InvestmentPlanning(hours=2*24, budget=450, timelimit=200)
    # Build model
    ip.build_model()
    # Run optimization
    ip.run()
    # Display results
    ip.display_results()
#%%