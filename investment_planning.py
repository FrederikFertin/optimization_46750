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

    def _add_lower_level_constraints(self):
        
        # Fix this, as this is our own production. Need variables for other generators
        # self.variables.p_g = {g: {t: self.model.addVar(lb=0, name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}

        # Define variables of lower level KKTs
        self.variables.lmd = {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='spot price at time {0}'.format(t)) for t in self.TIMES}
        self.variables.p_d = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.mu_under = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}
        self.variables.mu_over = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}
        self.variables.sigma_under = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.sigma_over = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}

        self.model.update()

        # Add lower level constraints. Rewrite to this format
        # gen_under = m.addConstrs((-p_G[g,t] * mu_under[g,t] == 0 for g in range(G) for t in range(T)), name = "balance_comp")
        # gen_upper = m.addConstrs(((p_G[g,t] - P_bar[g,t]) * mu_over[g,t] == 0 for g in range(G) for t in range(T)), name = "gen_upper")
        # dem_under = m.addConstrs((-p_D[d,t] * sigma_under[d,t] == 0 for d in range(D) for t in range(T)), name = "dem_under")
        # dem_upper = m.addConstrs(((p_D[d,t] - D_bar[d,t]) * sigma_over[d,t] == 0 for d in range(D) for t in range(T)), name = "dem_upper")

        # balance = m.addConstrs((sum(p_G[g,t] for g in range(G)) - sum(p_D[d,t] for d in range(D)) == 0 for t in range(T)), name = "balance")

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
                            + 8760/self.n_hours * gb.quicksum(self.v_OPEX[g] * self.variables.p_g[g][t] for t in self.TIMES)
                            for g in self.TECHNOLOGIES)
        # Define revenue (sum of generation revenues)
        revenue = gb.quicksum(self.variables.lmd[t] * self.variables.p_g[g][t] for g in self.TECHNOLOGIES for t in self.TIMES)
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


