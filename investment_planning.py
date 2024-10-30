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
        self.BUDGET = 1000 # set budget for capital costs in M€
        self._build_model()

    def _add_lower_level_variables(self):
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.TECHNOLOGIES

        # Define variables of lower level KKTs
        self.variables.p_g = {g: {t: self.model.addVar(lb=0, name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.GENERATORS}
        self.variables.p_g = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.p_d = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.lmd = {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='spot price at time {0}'.format(t)) for t in self.TIMES}
        
        self.variables.mu_under    = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.mu_over     = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.PRODUCTION_UNITS}
        self.variables.sigma_under = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.sigma_over  = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}

    def _add_lower_level_constraints(self):
        # Implement eq 13 + 14
        self.constraints.gen_under = self.model.addConstrs((-self.variables.p_g[g][t] * self.variables.mu_under[g][t] == 0 for g in self.PRODUCTION_UNITS for t in self.TIMES), name = "gen_under")

        # Fix P_bar and create seperate for our production units
        self.constraints.gen_upper = self.model.addConstrs(((self.variables.p_g[g][t] - self.P_bar[g][t]) * self.variables.mu_over[g][t] == 0 for g in self.PRODUCTION_UNITS for t in self.TIMES), name = "gen_upper")
        self.constraints.dem_under = self.model.addConstrs((-self.variables.p_d[d][t] * self.variables.sigma_under[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_under")
        self.constraints.dem_upper = self.model.addConstrs(((self.variables.p_d[d][t] - self.D_bar[d][t]) * self.variables.sigma_over[d][t] == 0 for d in self.DEMANDS for t in self.TIMES), name = "dem_upper")
        self.constraints.balance   = self.model.addConstrs((sum(self.variables.p_g[g][t] for g in self.PRODUCTION_UNITS) - sum(self.variables.p_d[d][t] for d in self.DEMANDS) == 0  for t in self.TIMES), name = "balance")

    def _build_model(self):
        self.model = gb.Model(name='Investment Planning')

        """ Initialize variables """
        # Investment in generation technologies (in MW)
        self.variables.P_investment = {g : self.model.addVar(lb=0, ub=GRB.INFINITY, name='investment in {0}'.format(g)) for g in self.TECHNOLOGIES}
        # Get lower level variables
        self._add_lower_level_variables()

        self.model.update()

        """ Initialize objective to maximize NPV [M€] """
        # Define costs (annualized capital costs + fixed and variable operational costs)
        costs = gb.quicksum( self.variables.P_investment[g] * (self.AF[g] * self.CAPEX[g] + self.f_OPEX[g])
                            + 8760/self.T * gb.quicksum(self.v_OPEX[g] * self.variables.p_g[g][t] for t in self.TIMES)
                            for g in self.TECHNOLOGIES)
        # Define revenue (sum of generation revenues)
        revenue = 8760/self.T * gb.quicksum(self.variables.lmd[t] * self.variables.p_g[g][t] for g in self.TECHNOLOGIES for t in self.TIMES)
        # Define NPV
        npv = revenue - costs
        # Set objective
        self.model.setObjective(npv, gb.GRB.MAXIMIZE)

        self.model.update()

        """ Initialize constraints """
        # Budget constraints
        self.constraints.budget = self.model.addConstr(gb.quicksum(
                                    self.variables.P_investment[g] * self.CAPEX[g] for g in self.TECHNOLOGIES)
                                    <= self.BUDGET, name='budget')

        self._add_lower_level_constraints()


    def run(self):
        self.model.optimize()
        self._save_data()
    
    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save investment values
        self.data.investment_values = {g : self.variables.P_investment[g].x for g in self.TECHNOLOGIES}
        
        # Save generation dispatch values
        self.data.generation_dispatch_values = {(g,t) : self.variables.p_g[g][t].x for g in self.TECHNOLOGIES for t in self.TIMES}
        
        # Save uniform prices lambda
        self.data.lambda_ = {t : self.variables.lmd[t].x for t in self.TIMES}

    def display_results(self):
        print('Maximal NPV: {0}'.format(self.data.objective_value))
        print('Investment Values: {0}'.format(self.data.investment_values))


if __name__ == '__main__':
    ip = InvestmentPlanning()
    ip.run()
    ip.display_results()