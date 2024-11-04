import gurobipy as gb
from network import Network
from gurobipy import GRB

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class InvestmentPlanning(Network):
    
    def __init__(self): # initialize class
        super().__init__()
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        self.BUDGET = 1000 * 10**6 # set budget for capital costs in € (1 billion €)
        self.TIMES = [self.TIMES[0]]
        self.T = len(self.TIMES)
        self._build_model()

    def _add_lower_level_variables(self):
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS

        # Define variables of lower level KKTs
        self.variables.p_g = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.p_d = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.lmd = {t: self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='spot price at time {0}'.format(t)) for t in self.TIMES} # Hourly spot price (€/MWh)
        
        self.variables.mu_under    = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over     = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.sigma_over  = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}

    def _add_lower_level_constraints(self):
        # KKT for lagrange objective derived wrt. generation variables
        self.constraints.gen_lagrange_generators = self.model.addConstrs((self.C_G_offer[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.GENERATORS for t in self.TIMES), name = "derived_lagrange_generators")
        self.constraints.gen_lagrange_windturbines = self.model.addConstrs((self.v_OPEX['Offshore Wind'] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.WINDTURBINES for t in self.TIMES), name = "derived_lagrange_windturbines")
        self.constraints.gen_lagrange_investments = self.model.addConstrs((self.v_OPEX[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t] == 0 for g in self.INVESTMENTS for t in self.TIMES), name = "derived_lagrange_investments")
        # KKT for lagrange objective derived wrt. demand variables
        self.constraints.dem_lagrange = self.model.addConstrs((self.U_D[t][d] - self.variables.lmd[t] - self.variables.sigma_under[d][t] + self.variables.sigma_over[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "derived_lagrange_demand")
        # KKT for generation minimal production
        self.constraints.gen_under = self.model.addConstrs((-self.variables.p_g[g][t] * self.variables.mu_under[g][t] == 0 for g in self.PRODUCTION_UNITS for t in self.TIMES), name = "gen_under")
        # KKT for generation capacities
        self.constraints.gen_upper_generators = self.model.addConstrs(((self.variables.p_g[g][t] - self.P_G_max[g]) * self.variables.mu_over[g][t] == 0 for g in self.GENERATORS for t in self.TIMES), name = "gen_upper_generators")
        self.constraints.gen_upper_windturbines = self.model.addConstrs(((self.variables.p_g[g][t] - self.P_W[t][g]) * self.variables.mu_over[g][t] == 0 for g in self.WINDTURBINES for t in self.TIMES), name = "gen_upper_windturbines")
        self.constraints.gen_upper_investments = self.model.addConstrs(((self.variables.p_g[g][t] - self.variables.P_investment[g]) * self.variables.mu_over[g][t] == 0 for g in self.INVESTMENTS for t in self.TIMES), name = "gen_upper_investments")
        # Generation capacity limits
        self.constraints.gen_cap_generators = self.model.addConstrs((self.variables.p_g[g][t] <= self.P_G_max[g] for g in self.GENERATORS for t in self.TIMES), name = "gen_cap_generators")
        self.constraints.gen_cap_windturbines = self.model.addConstrs((self.variables.p_g[g][t] <= self.P_W[t][g] for g in self.WINDTURBINES for t in self.TIMES), name = "gen_cap_windturbines")
        self.constraints.gen_cap_investments = self.model.addConstrs((self.variables.p_g[g][t] <= self.variables.P_investment[g] for g in self.INVESTMENTS for t in self.TIMES), name = "gen_cap_investments")
        # KKT for demand constraints
        self.constraints.dem_under = self.model.addConstrs((-self.variables.p_d[d][t] * self.variables.sigma_under[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_upper = self.model.addConstrs(((self.variables.p_d[d][t] - self.P_D[t][d]) * self.variables.sigma_over[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_upper")
        # Demand magnitude constraints
        self.constraints.dem_mag = self.model.addConstrs((self.variables.p_d[d][t] <= self.P_D[t][d] for d in self.DEMANDS for t in self.TIMES), name = "dem_mag")
        # KKT for balancing constraint
        self.constraints.balance = self.model.addConstrs((sum(self.variables.p_g[g][t] for g in self.PRODUCTION_UNITS) - sum(self.variables.p_d[d][t] for d in self.DEMANDS) == 0  for t in self.TIMES), name = "balance")

    def _build_model(self):
        self.model = gb.Model(name='Investment Planning')

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
        # Define revenue (sum of generation revenues)
        revenue = 20 * 8760/self.T * gb.quicksum(self.variables.lmd[t] * self.variables.p_g[g][t] for g in self.INVESTMENTS for t in self.TIMES)
        # Define NPV
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
    
    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save investment values
        self.data.investment_values = {g : self.variables.P_investment[g].x for g in self.INVESTMENTS}
        
        # Save generation dispatch values
        self.data.generation_dispatch_values = {(g,t) : self.variables.p_g[g][t].x for g in self.INVESTMENTS for t in self.TIMES}
        
        # Save uniform prices lambda
        self.data.lambda_ = {t : self.variables.lmd[t].x for t in self.TIMES}

    def display_results(self):
        print('Maximal NPV: {0}'.format(self.data.objective_value))
        print('Investment Values: {0}'.format(self.data.investment_values))


if __name__ == '__main__':
    ip = InvestmentPlanning()
    ip.run()
    ip.display_results()