import gurobipy as gb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from iterative_ip import expando
from network import Network
import pandas as pd
from common_methods import CommonMethods

# Confirmed the same as the other NodalCLearing:
class NodalClearing(Network, CommonMethods):
    
    def __init__(self,
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
        
        self.chosen_hours = chosen_hours
        self.T = len(chosen_hours) # set number of hours in optimization
        self.carbontax = carbontax # set carbon tax in €/tCO2
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)
        self.root_node = 'N13'
        
        self._initialize_fluxes_demands()
        self._initialize_costs()

        self.P_investment = P_investment
        if P_investment is None:
            self.P_investment = {i: {n: 0 for n in self.node_I[i]} for i in self.INVESTMENTS}

    def build_model(self):
        self.model = gb.Model(name='Nodal clearing')

        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        self.variables.p_g          = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d          = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta        = {n: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.NODES}

        """ Initialize objective function """
        self.model.setObjective(gb.quicksum(self.U_D[d] * self.variables.p_d[d][n][t] for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES)
                               -gb.quicksum(self.C_offer[g] * self.variables.p_g[g][n][t] for g in self.PRODUCTION_UNITS for n in self.node_production[g] for t in self.TIMES)
                                , gb.GRB.MAXIMIZE)
        self.model.update()

        """ Initialize constraints """
        # Generation capacity limits:
        self.constraints.p_gen_cap  = self.model.addConstrs((self.variables.p_g[g][n][t] <= self.P_G_max[g]
                                                            for g in self.GENERATORS for t in self.TIMES for n in self.node_G[g]), name = "gen_cap_generators")
        self.constraints.p_wt_cap   = self.model.addConstrs((self.variables.p_g[w][n][t] <= self.P_W[t][w]
                                                            for w in self.WINDTURBINES for t in self.TIMES for n in self.node_W[w]), name = "gen_cap_windturbines")
        self.constraints.p_inv_cap  = self.model.addConstrs((self.variables.p_g[i][n][t] <= self.P_investment[i][n] * self.fluxes[i][t] * self.cf[i]
                                                            for i in self.INVESTMENTS for t in self.TIMES for n in self.node_I[i]), name = "gen_cap_investments")
        # Demand magnitude constraints:
        self.constraints.demand_cap = self.model.addConstrs((self.variables.p_d[d][n][t] <= self.P_D[t][d]
                                                            for d in self.DEMANDS for n in self.node_D[d] for t in self.TIMES), name = "dem_mag")
        # Balancing constraint for each node:
        self.constraints.balance    = self.model.addConstrs(( gb.quicksum(self.variables.p_d[d][n][t]   for d in self.map_d[n])
                                                            - gb.quicksum(self.variables.p_g[g][n][t]   for g in self.map_g[n])
                                                            - gb.quicksum(self.variables.p_g[w][n][t]   for w in self.map_w[n])
                                                            - gb.quicksum(self.variables.p_g[i][n][t]   for i in self.map_i[n])
                                                            + gb.quicksum(self.L_susceptance[self.map_n[n][m]] * (self.variables.theta[n][t] - self.variables.theta[m][t])
                                                                         for m in self.map_n[n].keys()) == 0
                                                            for n in self.NODES for t in self.TIMES), name = "balance")
        # Line capacity constraints:
        
        self.constraints.line_l_cap = self.model.addConstrs((self.L_susceptance[self.map_n[n][m]] * (self.variables.theta[n][t] - self.variables.theta[m][t]) >= -self.L_cap[self.map_n[n][m]]
                                                            for n in self.NODES for m in self.map_n[n].keys() for t in self.TIMES), name = "line_cap_lower")
        self.constraints.line_u_cap = self.model.addConstrs((self.L_susceptance[self.map_n[n][m]] * (self.variables.theta[n][t] - self.variables.theta[m][t]) <= self.L_cap[self.map_n[n][m]]
                                                            for n in self.NODES for m in self.map_n[n].keys() for t in self.TIMES), name = "line_cap_lower")
        # Reference voltage angle:
        self.constraints.ref_angle  = self.model.addConstrs((self.variables.theta[self.root_node][t] == 0
                                                            for t in self.TIMES), name = "ref_angle")
        
        self.model.update()

    def _save_data(self):
        # Save objective value
        self.data.objective_value = self.model.ObjVal
        
        # Save dispatch values
        self.data.generator_dispatch = {g : {n : {t : self.variables.p_g[g][n][t].x for t in self.TIMES} for n in self.node_G[g]} for g in self.GENERATORS}
        self.data.demand_dispatch = {d : {t : self.variables.p_d[d][self.map_d[d][0]][t].x for t in self.TIMES} for d in self.DEMANDS}

        # Save generation dispatch values
        self.data.investment_dispatch = {g : {n : {t : self.variables.p_g[g][n][t].x for t in self.TIMES} for n in self.node_I[g]} for g in self.INVESTMENTS}
        
        # Save uniform prices lambda
        self.data.lambda_ = {n : {t : self.constraints.balance[n,t].pi for t in self.TIMES} for n in self.NODES}

        # Save voltage angles
        self.data.theta = {n : {t : self.variables.theta[n][t].x for t in self.TIMES} for n in self.NODES}

        self.costs = sum(self.P_investment[g][n] * (self.AF[g] * self.CAPEX[g] + self.f_OPEX[g])
                            + 8760/self.T * sum(self.v_OPEX[g] * self.data.investment_dispatch_values[g][n][t] for t in self.TIMES)
                            for g in self.INVESTMENTS for n in self.node_I[g])
        # Define revenue (sum of generation revenues) [M€]
        self.revenue = (8760 / self.T / 10**6) * sum(self.cf[g] * 
                                            self.data.lambda_[n][t] * self.data.investment_dispatch_values[g][n][t]
                                            for g in self.INVESTMENTS for t in self.TIMES for n in self.node_I[g])
        
        # Define NPV
        self.data.npv = self.revenue - self.costs

    def run(self):
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        self._save_data()

class NodalClearingDecomposed(Network, CommonMethods):
    
    def __init__(self,
                 node:str,
                 chosen_hours:list[str] = ['T1', 'T2', 'T3', 'T4', 'T5',],
                 timelimit:float = 100,
                 carbontax:float = 50,
                 seed:int = 42,
                 P_investment = None,
                 output_flag:int = 0,
                ): # initialize class
        super().__init__()

        np.random.seed(seed)

        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes

        # Define node subproblem
        self.node = node
        self.connected_nodes = list(self.map_n[self.node].keys())
        self.considered_nodes = self.connected_nodes + [self.node] 
        
        self.chosen_hours = chosen_hours
        self.T = len(chosen_hours) # set number of hours in optimization
        self.TIMES = chosen_hours
        self.carbontax = carbontax # set carbon tax in €/tCO2
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)
        self.root_node = 'N13'
        self.output_flag = output_flag
        
        self._initialize_fluxes_demands()
        self._initialize_costs()

        self.P_investment = P_investment
        if P_investment is None:
            self.P_investment = {i: {n: 0 for n in self.node_I[i]} for i in self.INVESTMENTS}

    def set_objective(self, lambda_, gamma, theta_hat):        
        self.model.setObjective(gb.quicksum(
                               - gb.quicksum(self.U_D[d] * self.variables.p_d[d][n][t] for d in self.DEMANDS for n in self.node_D[d] if n == self.node)
                               + gb.quicksum(self.C_offer[g] * self.variables.p_g[g][n][t] for g in self.PRODUCTION_UNITS for n in self.node_production[g] if n == self.node)
                               + gb.quicksum(lambda_[self.node][m][t] * self.variables.theta[m][t] for m in self.considered_nodes)
                               + gamma / 2 * gb.quicksum((self.variables.theta[m][t] - theta_hat[m][t])**2 for m in self.considered_nodes)
                                 for t in self.TIMES), gb.GRB.MINIMIZE)
                            
        self.model.update()

    def build_model(self):
        self.model = gb.Model(name='Nodal clearing')
        self.model.Params.OutputFlag = self.output_flag # set output flag
        self.model.Params.TIME_LIMIT = self.timelimit # set time limit for optimization to 100 seconds
        self.model.Params.Seed = 42 # set seed for reproducibility

        """ Initialize variables """
        self.variables.p_g          = {g: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='generation from {0} at node {1} at time {2}'.format(g,n,t)) for t in self.TIMES if n == self.node} for n in self.node_production[g]} for g in self.PRODUCTION_UNITS}
        self.variables.p_d          = {d: {n: {t:   self.model.addVar(lb=0,             ub=GRB.INFINITY, name='demand from {0} at node {1} at time {2}'.format(d, n, t)) for t in self.TIMES if n == self.node} for n in self.node_D[d]} for d in self.DEMANDS}
        self.variables.theta        = {n: {t:       self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_{0}_{1}'.format(n, t)) for t in self.TIMES} for n in self.considered_nodes}

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
                                                            for d in self.DEMANDS for t in self.TIMES for n in self.node_D[d] if n == self.node), name = "dem_mag")
        # Balancing constraint for each node:
        self.constraints.balance    = self.model.addConstrs((
                                                             gb.quicksum(self.variables.p_d[d][self.node][t]   for d in self.map_d[self.node])
                                                           - gb.quicksum(self.variables.p_g[g][self.node][t]   for g in self.map_g[self.node])
                                                           - gb.quicksum(self.variables.p_g[w][self.node][t]   for w in self.map_w[self.node])
                                                           - gb.quicksum(self.variables.p_g[i][self.node][t]   for i in self.map_i[self.node])
                                                           + gb.quicksum(self.L_susceptance[self.map_n[self.node][m]] * (self.variables.theta[self.node][t] - self.variables.theta[m][t])
                                                                         for m in self.connected_nodes) == 0
                                                           for t in self.TIMES)
                                                           , name = "balance")
        # Line capacity constraints:
        self.constraints.line_l_cap = self.model.addConstrs((self.L_susceptance[self.map_n[self.node][m]] * (self.variables.theta[self.node][t] - self.variables.theta[m][t]) >= -self.L_cap[self.map_n[self.node][m]]
                                                            for m in self.connected_nodes for t in self.TIMES), name = "line_cap_lower")
        self.constraints.line_u_cap = self.model.addConstrs((self.L_susceptance[self.map_n[self.node][m]] * (self.variables.theta[self.node][t] - self.variables.theta[m][t]) <= self.L_cap[self.map_n[self.node][m]]
                                                            for m in self.connected_nodes for t in self.TIMES), name = "line_cap_lower")
        # Reference voltage angle:
        if self.root_node in self.considered_nodes:
            self.constraints.ref_angle  = self.model.addConstrs((self.variables.theta[self.root_node][t] == 0
                                                            for t in self.TIMES), name = "ref_angle")
        
        self.model.update()

    def _get_theta_i(self):
        # Save voltage angles
        self.data.theta_i = {m : {t : self.variables.theta[m][t].x for t in self.TIMES} for m in self.considered_nodes}

    def solve(self):
        self.model.optimize()
        self.model.update()
        self._get_theta_i()

class ADMM():
    def __init__(self, models, lambdas, theta_i, theta_hat, K, tolerance, gamma = 1, tau = 1.1, nodes = None, considered_nodes = None, chosen_hours = None, network_instance = None):
        self.models = models
        self.lambdas = lambdas
        self.tolerance = tolerance
        self.gamma = gamma
        self.K = K
        self.theta_i = theta_i
        self.theta_hat = theta_hat
        self.subproblems = nodes
        self.nodes = nodes
        self.considered_nodes = considered_nodes
        self.TIMES = chosen_hours
        self.tau = tau
        self.primal_residuals = []
        self.dual_residuals = []
        self.gammas = []
        self.beta = 1
        self.alpha = 1
        self.betas = []
        self.alphas = []
        self.primal_scaler = sum(1 for n in self.subproblems for _ in self.considered_nodes[n] for _ in self.TIMES)
        self.dual_scaler = sum(1 for _ in self.nodes for _ in self.TIMES)
        self.network = network_instance

    def _update_theta_i(self):
        new_theta_i = {}
        for n in self.subproblems:
            self.models[n].set_objective(self.lambdas[-1], self.gamma, self.theta_hat[-1])
            self.models[n].solve()
            new_theta_i[n] = self.models[n].data.theta_i

        self.theta_i.append(new_theta_i)


    def _update_theta_hat(self):
        t_i = self.theta_i[-1]
        t_h_old = self.theta_hat[-1]
        # self.hat_model.setObjective(gb.quicksum(
        #         self.gamma/2 * (t_i[n][m][t] - self.var__t_h[m,t])**2
        #         for n in self.subproblems for m in self.considered_nodes[n] for t in self.TIMES), gb.GRB.MINIMIZE)
        # self.hat_model.optimize()
        
        # self.theta_hat.append({n : {t : self.var__t_h[n, t].x for t in self.TIMES} for n in self.nodes})
        t_h_new = {}
        for n in self.nodes:
            t_h_new[n] = {}
            for t in self.TIMES:
                tot = sum(t_i[m][n][t] for m in self.considered_nodes[n])
                size_ = sum(1 for _ in self.considered_nodes[n])
                t_h = tot / size_
                t_h_new[n][t] = self.alpha * t_h + (1 - self.alpha) * t_h_old[n][t]
        self.theta_hat.append(t_h_new)


    def _update_lambdas(self):
        new_lambdas = {}
        old_lambdas = self.lambdas[-1]
        t_i = self.theta_i[-1]
        t_hat = self.theta_hat[-1]
        for n in self.subproblems:
            new_lambdas[n] = {}
            for m in self.considered_nodes[n]:
                new_lambdas[n][m] = {}
                for t in self.TIMES:
                    n_l = old_lambdas[n][m][t] + self.gamma * (t_i[n][m][t] - t_hat[m][t])
                    new_lambdas[n][m][t] = n_l * self.alpha + old_lambdas[n][m][t] * (1 - self.alpha)
        
        self.lambdas.append(new_lambdas)

    # def _update_lambdas(self):
    #     new_lambdas = {}
    #     old_lambdas = self.lambdas[-1]
    #     t_i = self.theta_i[-1]
    #     t_hat = self.theta_hat[-1]
    #     for m in self.nodes:
    #         new_lambdas[n] = {}
    #         for t in self.TIMES:
    #             new_lambdas[n][t] = old_lambdas[n][m][t] + self.gamma * sum(t_i[n][m][t] - t_hat[m][t] for m in self.considered_nodes[n])
        
    #     self.lambdas.append(new_lambdas)

    def _check_convergence(self):
        t_hat_current = self.theta_hat[-1]
        t_hat_previous = self.theta_hat[-2]
        dual_residual = self.gamma * np.sqrt(sum((t_hat_current[n][t] - t_hat_previous[n][t])**2 for n in self.nodes for t in self.TIMES)) / self.dual_scaler
        lambda_current = self.lambdas[-1]
        lambda_previous = self.lambdas[-2]
        primal_residual = np.sqrt(sum((lambda_current[n][m][t] - lambda_previous[n][m][t])**2 for n in self.subproblems for m in self.considered_nodes[n] for t in self.TIMES)) / self.primal_scaler

        print(f'Primal residual: {primal_residual}')
        print(f'Dual residual: {dual_residual}')
        if primal_residual > dual_residual:
            self.gamma = self.tau * self.gamma
        else:
            self.gamma = self.gamma / self.tau
        if not(self.tau == 1):
            print(f'Gamma: {self.gamma}')

        if len(self.primal_residuals) > 0:
            if max(primal_residual, dual_residual)/max(self.primal_residuals[-1], self.dual_residuals[-1]) < 1:
                self.beta_new = (1 + np.sqrt(1 + 4 * self.beta**2)) / 2
                self.alpha = 1 - (self.beta - 1) / self.beta_new
                self.beta = self.beta_new
            else:
                self.alpha = 1
        self.betas.append(self.beta)
        self.alphas.append(self.alpha)
        self.primal_residuals.append(primal_residual)
        self.dual_residuals.append(dual_residual)
        self.gammas.append(self.gamma)

        if primal_residual > self.tolerance or dual_residual > self.tolerance:
            return False
        
        return True
    
    def _algorithm(self):
        self.hat_model = gb.Model(name='Theta_hat_update')
        self.hat_model.Params.OutputFlag = 0
        self.var__t_h = self.hat_model.addVars(self.nodes, self.TIMES, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta_hat')

        for k in range(self.K):
            #self.gamma = np.sqrt(k+1)
            # Solve the subproblems
            self._update_theta_i()
            # Update theta_hat
            self._update_theta_hat()
            # Update lambdas
            self._update_lambdas()
            # Check for convergence
            print()
            print(f'Iteration {k}')
            if self._check_convergence():
                return k
        return self.K
        
    def _save_results(self):
        self.demand_dispatch = {d : { t: round(self.models[self.network.node_D[d][0]].variables.p_d[d][self.network.node_D[d][0]][t].x,2) for t in self.TIMES} for d in self.network.DEMANDS}
        self.gen_dispatch = {g : { t: round(self.models[self.network.node_G[g][0]].variables.p_g[g][self.network.node_G[g][0]][t].x,2) for t in self.TIMES} for g in self.network.GENERATORS}

    def run_algorithm(self):
        self.iterations = self._algorithm()
        self._save_results()


if __name__ == "__main__":
    network_instance = Network()

    timelimit = 100
    carbontax = 60
    seed = 38
    chosen_days = range(1)
    chosen_hours = list('T{0}'.format(i+1) for d in chosen_days for i in range(d*24, (d+1)*24))
    chosen_hours = ['T1',]

    market_clearing_instances = {}
    # For each subproblem/node, denote nodes that are considered in each subproblem
    considered_nodes = {node : list(network_instance.map_n[node].keys()) + [node] for node in network_instance.NODES}
    # Initialize lambdas to 0 for each subproblem and each time period
    #lambdas = [{subproblem: {t : 0 for t in chosen_hours} for subproblem in network_instance.NODES}]
    lambdas = [{subproblem: {considered_node : {t : 0 for t in chosen_hours} for considered_node in considered_nodes[subproblem]} for subproblem in network_instance.NODES}]
    # Initialize theta_hat to 0 for each node and each time period:
    theta_hat = [{node : {t : 0 for t in chosen_hours} for node in network_instance.NODES}]
    # Initialize copy variables of theta_i (variables in complicating constraints) to 0 for each subproblem and each considered node and each time period:
    theta_i = [{subproblem: {considered_node : {t : 0 for t in chosen_hours} for considered_node in considered_nodes[subproblem]} for subproblem in network_instance.NODES}]
    
    gamma = 10 # Gradient step size
    tau = 1.0 # Step size multiplier
    K = 1000 # Maximum number of iterations

    # Create a submodel for each generator
    print()
    print('Creating models for each node:')
    for n in network_instance.NODES:
        print(f'Node {n}')
        market_clearing_instances[n] = NodalClearingDecomposed(node=n, chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed)
        market_clearing_instances[n].build_model()
    
    nc = NodalClearing(chosen_hours=chosen_hours, timelimit=timelimit, carbontax=carbontax, seed=seed) # For comparing the results
    nc.build_model()
    nc.run()

    # Create ADMM instance
    admm = ADMM(models=market_clearing_instances, lambdas=lambdas, theta_hat=theta_hat, theta_i=theta_i, K=K, tolerance=0.01, tau = tau, gamma=gamma, nodes = network_instance.NODES, considered_nodes = considered_nodes, chosen_hours = chosen_hours, network_instance = network_instance)
    admm.run_algorithm()

    # Plot primal and dual residuals and gamma
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(admm.primal_residuals, label='Primal residual')
    ax.plot(admm.dual_residuals, label='Dual residual')
    #ax2 = ax.twinx()
    #ax2.plot(admm.gammas, label='Gamma', color='green')
    #ax2.set_ylabel('Gamma')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(Residual)')
    ax.set_yscale('log')
    ax.set_title('ADMM convergence, ' + r'$\gamma$' + f' = {admm.gamma}')
    # Get handles and labels for a legend
    lines, labels = ax.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax.legend(lines, labels, loc='upper right')
    plt.show()


    # Plot primal and dual residuals and alpha for the accelerated version
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(admm.primal_residuals, label='Primal residual')
    ax.plot(admm.dual_residuals, label='Dual residual')
    ax2 = ax.twinx()
    ax2.plot(admm.alphas, label=r'$\alpha$', color='green')
    #ax2.plot(admm.betas, label=r'$\beta$', color='red')
    ax2.set_ylabel(r'$\alpha$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log(Residual)')
    ax.set_yscale('log')
    ax.set_title('ADMM convergence, ' + r'$\gamma$' + f' = {admm.gamma} - accelerated')
    # Get handles and labels for a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax2.grid(False)
    #ax.legend(lines, labels, loc='upper right')
    plt.show()
    # Compare the results with the centralized model
    

    print()