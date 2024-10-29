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
        self._build_model()

    def _add_lower_level_constraints(self):
        # Define variables of lower level KKTs
        self.variables.lmd = {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='spot price at time {0}'.format(t)) for t in self.TIMES}
        self.variables.p_g = {g: {t: self.model.addVar(lb=0, ub=self.P_G_max[g], name='generation from {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}
        self.variables.p_d = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='demand from {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.mu_under = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}
        self.variables.mu_over = {g: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on generator {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}
        self.variables.sigma_under = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for lb on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}
        self.variables.sigma_over = {d: {t: self.model.addVar(lb=0, ub=GRB.INFINITY, name='Dual for ub on demand {0} at time {1}'.format(d, t)) for t in self.TIMES} for d in self.DEMANDS}

        self.model.update()

        # Add lower level constraints
        self.constraints.gen_constr = {g: {t: self.model.addConstr(self.C_G_offer[g] - self.variables.lmd[t] - self.variables.mu_under[g][t] + self.variables.mu_over[g][t],
                                                                    gb.GRB.LESS_EQUAL, self.P_G_max[g], name='Generation upper bound for {0} at time {1}'.format(g, t)) for t in self.TIMES} for g in self.TECHNOLOGIES}

    def _build_model(self):
        self.model = gb.Model(name='Investment Planning')

        # Initialize variables
        self.variables.P_investment = {g:self.model.addVar(lb=0, ub=GRB.INFINITY, name='investment in {0}'.format(g)) for g in self.TECHNOLOGIES}

        self._add_lower_level_variables()

        self.model.update()

        # initialize objective to maximize NPV
        costs = gb.quicksum( self.variables.P_investment[g] * (self.AF[g] * self.CAPEX[g] + self.f_OPEX[g])
                            + 8760/self.n_hours * gb.quicksum(self.v_OPEX[g] * self.variables.p_g[g][t] for t in self.TIMES)
                            for g in self.TECHNOLOGIES)
        
        revenue = gb.quicksum(self.variables.lmd[t] * self.variables.p_g[g][t] for g in self.TECHNOLOGIES for t in self.TIMES)
        npv = revenue - costs
        self.model.setObjective(npv, gb.GRB.MAXIMIZE)

        self.model.update()

        # initialize constraints

        # Budget constraints

        self._add_lower_level_constraints()



