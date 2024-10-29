import gurobipy as gb
from network import Network
from methods import CommonMethods

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class EconomicDispatch(Network, CommonMethods):
    
    def __init__(self, n_hours: int, ramping: bool, battery: bool, hydrogen: bool): # initialize class
        super().__init__()
        
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        self.TIMES = self.TIMES[:n_hours] # set number of hours to run the model
        self.ramping = ramping # Is ramping constraints enforced?
        self.battery = battery # Is battery included in the model?
        self.H2 = hydrogen # Is hydrogen included in the model?
        if not battery: 
            self.BATTERIES = []
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables 
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.H2:
            # Place one electrolyzer at each wind farm
            self.variables.hydrogen = {(w,t):self.model.addVar(lb=0,ub=100,name='consumption of electrolyzer {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.battery:
            # Initialize battery variables with lower and upper bounds on capacity and power
            self.variables.battery_soc = {(b,t):self.model.addVar(lb=0,ub=self.batt_cap[b],name='soc of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_ch = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='dispatch of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
            self.variables.battery_dis = {(b,t):self.model.addVar(lb=0,ub=self.batt_power[b],name='consumption of battery {0}'.format(b)) for b in self.BATTERIES for t in self.TIMES}
        
        self.model.update()
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.U_D[t][d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)
        
        # initialize constraints
        # initialize constraints 
        ## Step 1 - getting duals for the marginal generator:
        #self.constraints.generation_constraint_min = {t:self.model.addConstr(
        #    self.variables.generator_dispatch['G7',t], gb.GRB.GREATER_EQUAL, 0.1, name='Min gen G7') for t in self.TIMES}
        #self.constraints.generation_constraint_max = {t:self.model.addConstr(
        #    self.variables.generator_dispatch['G7',t], gb.GRB.LESS_EQUAL, self.P_G_max['G7']-0.1, name='Max gen G7') for t in self.TIMES}
        
        # Balance constraints
        # Evaluates based on the values of self.battery and self.H2
        self.constraints.balance_constraint = self._add_balance_constraints()
        
        # ramping constraints
        if self.ramping:
            self._add_ramping_constraints()

        # battery constraints
        if self.battery:
            self._add_battery_constraints()
        
        # electrolyzer constraints
        if self.H2:
            self._add_hydrogen_constraints()
        
    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {(d,t):self.variables.consumption[d,t].x for d in self.DEMANDS for t in self.TIMES}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save wind turbine dispatches 
        self.data.wind_dispatch_values = {(w,t):self.variables.wind_turbines[w,t].x for w in self.WINDTURBINES for t in self.TIMES}

        # save up and down regulation constraints        
        if self.ramping:
            self.data.ramping_up_dual = {t : {g:self.constraints.ramping_up[g,t].Pi for g in self.GENERATORS} for t in self.TIMES[1:]}
            self.data.ramping_dw_dual = {t : {g:self.constraints.ramping_dw[g,t].Pi for g in self.GENERATORS} for t in self.TIMES[1:]}

        # save battery dispatches
        if self.battery:
            self.data.battery = {(b,t):self.variables.battery_ch[b,t].x - self.variables.battery_dis[b,t].x for b in self.BATTERIES for t in self.TIMES}
            self.data.battery_soc = {(b,t):self.variables.battery_soc[b,t].x for b in self.BATTERIES for t in self.TIMES}
            self.data.battery_soc_constraint = {(b,t):self.constraints.batt_soc[b,t].Pi for t in self.TIMES[1:] for b in self.BATTERIES}
        
        # save electrolyzer activity
        if self.H2:
            self.data.hydrogen = {(w,t):self.variables.hydrogen[w,t].x for w in self.WINDTURBINES for t in self.TIMES}

        
        # save uniform prices lambda 
        self.data.lambda_ = {t:self.constraints.balance_constraint[t].Pi for t in self.TIMES}
        
    def run(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        # calculate profits of suppliers ( profits = (C_G - lambda) * p_G )
        self.results.profits_G = {g:sum((self.data.lambda_[t] - self.C_G_offer[g])  * self.data.generator_dispatch_values[g,t] for t in self.TIMES) for g in self.GENERATORS}
        self.results.profits_W = {w:sum(self.data.lambda_[t] * self.data.wind_dispatch_values[w,t] for t in self.TIMES) for w in self.WINDTURBINES}
        
        # calculate utility of suppliers ( (U_D - lambda) * p_D )
        self.results.utilities = {d:sum((self.U_D[t][d] - self.data.lambda_[t]) * self.data.consumption_values[d,t] for t in self.TIMES) for d in self.DEMANDS}

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Market clearing prices: " + str(self.data.lambda_))
        print()
        print("Social welfare: " + str(self.data.objective_value))
        print()
        print("Profit of suppliers: ")
        print("Generators: ")
        print(self.results.profits_G)
        print("Wind turbines: ")
        print(self.results.profits_W)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)
