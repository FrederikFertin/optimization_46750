import gurobipy as gb
from gurobipy import GRB
import numpy as np
from iterative_ip import NodalClearing, expando
from network import Network
import pandas as pd
from common_methods import CommonMethods

# Parameters

class NodalClearingDecomposed(Network, CommonMethods):
    
    def __init__(self,
                 node:str,
                 chosen_hours:list[str] = ['T1', 'T2', 'T3', 'T4', 'T5',],
                 timelimit:float = 100,
                 carbontax:float = 50,
                 seed:int = 42,
                 P_investment = None,
                ): # initialize class
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        self.node = node
        self.connected_nodes = list(self.map_n[self.node].keys())
        self.considered_nodes = self.connected_nodes + [self.node] 
        
        self.chosen_hours = chosen_hours
        self.T = len(chosen_hours) # set number of hours in optimization
        self.carbontax = carbontax # set carbon tax in €/tCO2
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)
        self.root_node = 'N1'
        
        self._initialize_fluxes_demands()
        self._initialize_costs()

        # Define generation costs
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS
        self.node_production = {**self.node_G, **self.node_I, **self.node_W}
        self.C_G_offer_modified = {g: round(self.C_G_offer[g] + (self.EF[g]*self.carbontax), 2) for g in self.GENERATORS} # Variable costs in €/MWh incl. carbon tax
        self.C_I_offer = {g: round(self.v_OPEX[g] * 10**6, 2) for g in self.INVESTMENTS} # Variable costs in €/MWh
        self.C_offer = {**self.C_G_offer_modified, **self.C_I_offer, **self.C_W_offer} # Indexed by PRODUCTION_UNITS

    def set_objective(self, lambda_, gamma, theta_hat):
        self.model.setObjective(-gb.quicksum(self.U_D[d] * self.variables.p_d[d][n][t] for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES if n == self.node)
                               + gb.quicksum(self.C_offer[g] * self.variables.p_g[g][n][t] for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES if n == self.node)
                               + gb.quicksum(lambda_[self.node][t] * self.variables.theta[n][t] for n in self.considered_nodes for t in self.TIMES)
                               + gamma * gb.quicksum((self.variables.theta[n][t] - theta_hat[n][t])**2 for n in self.considered_nodes for t in self.TIMES)
                                , gb.GRB.MINIMIZE)
        self.model.update()

    def build_model(self):
        self.model = gb.Model(name='Nodal clearing')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        self.variables.p_g          = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES if n == self.node} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d          = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES if n == self.node} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta        = {n: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.considered_nodes}

        self.model.update()

        """ Initialize constraints """
        # Generation capacity limits:
        self.constraints.p_gen_cap  = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g]
                                                            for g in self.GENERATORS for t in self.TIMES for n in self.node_G[g] if n == self.node), name = "gen_cap_generators")
        self.constraints.p_wt_cap   = self.model.addConstrs((self.variables.p_g[w][n][t] <= self.P_W[t][w]
                                                            for w in self.WINDTURBINES for t in self.TIMES for n in self.node_W[w] if n == self.node), name = "gen_cap_windturbines")
        self.constraints.p_inv_cap  = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.P_investment[i][n] * self.fluxes[i][t] * self.cf[i]
                                                            for i in self.INVESTMENTS for t in self.TIMES for n in self.node_I[i] if n == self.node), name = "gen_cap_investments")
        # Demand magnitude constraints:
        self.constraints.demand_cap = self.model.addConstrs((self.variables.p_d[d][n][t] <= self.P_D[t][d]
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES if n == self.node), name = "dem_mag")
        # Balancing constraint for each node:
        self.constraints.balance    = self.model.addConstrs(( gb.quicksum(self.variables.p_d[d][self.node][t]   for d in self.map_d[self.node])
                                                            - gb.quicksum(self.variables.p_g[g][self.node][t]   for g in self.map_g[self.node])
                                                            - gb.quicksum(self.variables.p_g[w][self.node][t]   for w in self.map_w[self.node])
                                                            - gb.quicksum(self.variables.p_g[i][self.node][t]   for i in self.map_i[self.node])
                                                            + gb.quicksum(self.L_susceptance[self.map_n[self.node][m]] * (self.variables.theta[self.node][t] - self.variables.theta[m][t])
                                                                          for m in self.map_n[self.node].keys()) == 0
                                                            for t in self.TIMES), name = "balance")
        # Line capacity constraints:
        self.constraints.line_l_cap = self.model.addConstrs((self.L_susceptance[self.map_n[self.node][m]] * (self.variables.theta[self.node][t] - self.variables.theta[m][t]) >= -self.L_cap[self.map_n[self.node][m]]
                                                            for m in self.connected_nodes for t in self.TIMES), name = "line_cap_lower")
        self.constraints.line_u_cap = self.model.addConstrs((self.L_susceptance[self.map_n[self.node][m]] * (self.variables.theta[self.node][t] - self.variables.theta[m][t]) <= self.L_cap[self.map_n[self.node][m]]
                                                            for m in self.connected_nodes for t in self.TIMES), name = "line_cap_lower")
        # Reference voltage angle:
        if self.root_node in self.considered_nodes:
            self.constraints.ref_angle  = self.model.addConstrs((  self.variables.theta[self.root_node][t] == 0
                                                            for t in self.TIMES), name = "ref_angle")
        
        self.model.update()

    def _get_theta_i(self):
        # Save voltage angles
        self.data.theta_i = {n : {t : self.variables.theta[n][t].x for t in self.TIMES} for n in self.considered_nodes}

    def solve(self):
        self.model.optimize()
        self._get_theta_i()

class ADMM():
    def __init__(self, models, lambdas, theta_i, theta_hat, K, tolerance, gamma = 0.1, nodes = None, considered_nodes = None, chosen_hours = None):
        self.models = models
        self.lambdas = lambdas
        self.tolerance = tolerance
        self.gamma = gamma
        self.K = K
        self.theta_i = theta_i
        self.theta_hat = theta_hat
        self.nodes = nodes
        self.considered_nodes = considered_nodes
        self.chosen_hours = chosen_hours

    def _update_theta_hat(self):
        t_i = self.theta_i[-1]
        hat_model = gb.Model(name='Theta_hat_update')
        hat_model.Params.OutputFlag = 0
        t_h = hat_model.addVars(self.nodes, self.chosen_hours, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_hat')
        hat_model.setObjective(self.gamma/2 * gb.quicksum(gb.quicksum((t_i[m][t] - t_h[m][t])**2 for m in self.considered_nodes[n] for t in self.chosen_hours) for n in self.nodes), gb.GRB.MINIMIZE)
        hat_model.optimize()

        self.theta_hat.append({n : {t : t_h[n, t].x for t in self.chosen_hours} for n in self.nodes})


    def _update_theta_i(self):
        new_theta_i = {}
        for n in range(self.nodes):
            self.models[n].set_objective(self.lambdas[-1], self.gamma, self.theta_hat[-1])
            self.models[n].optimize()
            new_theta_i[n] = self.models[n].data.theta_i

        self.theta_i.append(new_theta_i)

    def _update_lambdas(self):
        new_lambdas = {}
        old_lambdas = self.lambdas[-1]
        t_i = self.theta_i[-1]
        t_hat = self.theta_hat[-1]
        for n in self.nodes:
            new_lambdas[n] = {}
            for t in self.chosen_hours:
                new_lambdas[n][t] = old_lambdas[n][t] + self.gamma * sum((t_i[n][m][t] - t_hat[m][t]) for m in self.considered_nodes[n] for t in self.chosen_hours)
        
        self.lambdas.append(new_lambdas)

    def _check_convergence(self):
        t_i = self.theta_i[-1]
        t_hat = self.theta_hat[-1]
        if abs(sum((t_i[n][m][t] - t_hat[m][t]) for n in self.nodes for m in self.considered_nodes[n] for t in self.chosen_hours)) > self.tolerance:
            return False
        return True
    
    def _algorithm(self):
        for k in range(self.K):
            # Solve the subproblems
            self._update_theta_i()
            # Update theta_hat
            self._update_theta_hat()
            # Update lambdas
            self._update_lambdas()
            # Check for convergence
            if self._check_convergence():
                return k
        return self.K
        
    def run_algorithm(self):
        self.iterations = self._algorithm()


if __name__ == "__main__":
    network_instance = Network()

    timelimit = 100
    carbontax = 60
    seed = 38
    chosen_days = range(365)
    chosen_hours = list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24))
    market_clearing_instances = {}
    models = {}
    considered_nodes = {node : list(network_instance.map_n.keys()) + [node] for node in network_instance.NODES}
    lambdas = [{node : {t : 0 for t in chosen_hours} for node in network_instance.NODES}]
    gamma = 0.1
    theta_hat = [{node : {t : 0 for t in chosen_hours} for node in network_instance.NODES}]
    theta_i = [{node: {considered_node : {t : 0 for t in chosen_hours} for considered_node in considered_nodes[node]} for node in network_instance.NODES}]
    K = 1000

    # Create a submodel for each generator
    for n in network_instance.NODES:
        market_clearing_instances[n] = NodalClearingDecomposed(node=n, chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
        market_clearing_instances[n].build_model()
        market_clearing_instances[n].model.Params.OutputFlag = 0
        market_clearing_instances[n].model.update()
    
    # Create ADMM instance
    admm = ADMM(models=market_clearing_instances, lambdas=lambdas, theta_hat=theta_hat, theta_i=theta_i, K=K, tolerance=0.0001, gamma=gamma, nodes = network_instance.NODES, considered_nodes = considered_nodes, chosen_hours = chosen_hours)
    admm.run_algorithm()

    