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



