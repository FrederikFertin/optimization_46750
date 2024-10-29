import matplotlib.pyplot as plt
import gurobipy as gb
from network import Network
from methods import CommonMethods
from gurobipy import GRB

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class NodalMarketClearing(Network, CommonMethods):
    
    def __init__(self, model_type: str): # initialize class
        super().__init__()
        self.data = expando() # build data attributes
        self.variables = expando() # build variable attributes
        self.constraints = expando() # build constraint attributes
        self.results = expando() # build results attributes
        if model_type != 'nodal' and model_type != 'zonal':
            raise ValueError('Model type should be either nodal or zonal')
        else:
            self.type = model_type
        self._build_model() # build gurobi model
    
    def _build_model(self):
        # initialize optimization model
        self.model = gb.Model(name='Economic Dispatch')
        
        # initialize variables      
        self.variables.consumption = {(d,t):self.model.addVar(lb=0,ub=self.P_D[t][d],name='consumption of demand {0}'.format(d)) for d in self.DEMANDS for t in self.TIMES}
        self.variables.generator_dispatch = {(g,t):self.model.addVar(lb=0,ub=self.P_G_max[g],name='dispatch of generator {0}'.format(g)) for g in self.GENERATORS for t in self.TIMES}
        self.variables.wind_turbines = {(w,t):self.model.addVar(lb=0,ub=self.P_W[t][w],name='dispatch of wind turbine {0}'.format(w)) for w in self.WINDTURBINES for t in self.TIMES}
        if self.type == 'nodal':
            # Nodal voltage angle variables - node N1 is reference node
            self.variables.theta = {(n,t):self.model.addVar(lb=-GRB.INFINITY,name='voltage angle at node {0}'.format(n)) for n in self.NODES for t in self.TIMES}
        elif self.type == 'zonal':
            # Zoanl interconnector flow variables - capacities enforced as lb and ub
            self.variables.ic = {(ic,t):self.model.addVar(lb=-self.ic_cap[ic],ub=self.ic_cap[ic],name='interconnector flow {0}'.format(ic)) for ic in self.INTERCONNECTORS for t in self.TIMES}
        
        self.model.update()
        
        # initialize objective to maximize social welfare
        demand_utility = gb.quicksum(self.U_D[t][d] * self.variables.consumption[d,t] for d in self.DEMANDS for t in self.TIMES)
        generator_costs = gb.quicksum(self.C_G_offer[g] * self.variables.generator_dispatch[g,t] for g in self.GENERATORS for t in self.TIMES)
        objective = demand_utility - generator_costs
        self.model.setObjective(objective, gb.GRB.MAXIMIZE)
        
        # initialize constraints 
        
        # balance constraint
        if self.type == 'nodal':
            self.constraints.balance_constraint = self._add_nodal_balance_constraints()
        elif self.type == 'zonal':
            self.constraints.balance_constraint = self._add_zonal_balance_constraints()
        
        # Line capacity constraints
        if self.type == 'nodal':
            self._add_line_capacity_constraints()
        

    def _add_nodal_balance_constraints(self):
        balance_constraints = {}

        for t in self.TIMES: # Add one balance constraint for each hour
            balance_constraints[t] = {}
            for n in self.NODES:
                # Contribution of generators
                generator_expr = gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.map_g[n])

                # Contribution of demands
                demand_expr = gb.quicksum(self.variables.consumption[d,t] for d in self.map_d[n])
                
                # Contribution of wind farms
                wind_expr = gb.quicksum(self.variables.wind_turbines[w,t] for w in self.map_w[n])
                
                # Import of power through lines
                line_expr = gb.quicksum(self.L_susceptance[line] * (self.variables.theta[n,t] - self.variables.theta[m,t])
                                        for m, line in self.map_n[n].items())

                # Central balance constraint
                balance_constraints[t][n] = self.model.addLConstr(
                                            demand_expr - generator_expr - wind_expr + line_expr,
                                            gb.GRB.EQUAL,
                                            0, name='Balance equation')
            
        return balance_constraints
        
    def _add_zonal_balance_constraints(self):
        balance_constraints = {}

        for t in self.TIMES: # Add one balance constraint for each hour
            balance_constraints[t] = {}
            for z in self.ZONES:
                # Contribution of generators
                generator_expr = gb.quicksum(self.variables.generator_dispatch[g,t]
                                             for n in self.map_z[z] for g in self.map_g[n])

                # Contribution of demands
                demand_expr = gb.quicksum(self.variables.consumption[d,t]
                                          for n in self.map_z[z] for d in self.map_d[n])

                # Contribution of wind farms
                wind_expr = gb.quicksum(self.variables.wind_turbines[w,t]
                                        for n in self.map_z[z] for w in self.map_w[n])
                
                # Import of power through interconnectors
                ic_expr = gb.quicksum(self.variables.ic[ic,t]
                                      for ic in self.zonal[z]) * ((-1) if z == 'Z2' else 1) # direction of ic is towards zone Z2

                # Central balance constraint
                balance_constraints[t][z] = self.model.addLConstr(
                                            demand_expr - generator_expr - wind_expr + ic_expr,
                                            gb.GRB.EQUAL,
                                            0, name='Balance equation')
            
        return balance_constraints
    
    def _add_line_capacity_constraints(self):
        # Line capacity constraints - runs through each line twice, once for each direction.
        # Thus only the max capacity is enforced.
        self.constraints.lines = {(n,m,t): self.model.addLConstr(
                self.L_susceptance[line] * (self.variables.theta[n,t] - self.variables.theta[m,t]),
                gb.GRB.LESS_EQUAL,
                self.L_cap[line],
                name='Line limit') for n in self.NODES for t in self.TIMES for m, line in self.map_n[n].items()}
        self.constraints.theta_ref = {t : self.model.addLConstr(self.variables.theta['N1',t], gb.GRB.EQUAL, 0, name='reference angle') for t in self.TIMES}

    def _save_data(self):
        # save objective value
        self.data.objective_value = self.model.ObjVal
        
        # save consumption values 
        self.data.consumption_values = {(d,t):self.variables.consumption[d,t].x for d in self.DEMANDS for t in self.TIMES}
        
        # save generator dispatches 
        self.data.generator_dispatch_values = {(g,t):self.variables.generator_dispatch[g,t].x for g in self.GENERATORS for t in self.TIMES}
        
        # save wind turbine dispatches 
        self.data.wind_dispatch_values = {(w,t):self.variables.wind_turbines[w,t].x for w in self.WINDTURBINES for t in self.TIMES}
        
        # save uniform prices lambda 
        if self.type == 'nodal':
            self.data.lambda_ = {t:{n:self.constraints.balance_constraint[t][n].Pi for n in self.NODES} for t in self.TIMES}
            self.data.theta = {t:{n:self.variables.theta[n,t].x for n in self.NODES} for t in self.TIMES}
            self.data.loading = {t:{n: {m:self.constraints.lines[n,m,t].Pi for m in self.map_n[n].keys()} for n in self.NODES} for t in self.TIMES}
        elif self.type == 'zonal':
            self.data.lambda_ = {t:{z:self.constraints.balance_constraint[t][z].Pi for z in self.ZONES} for t in self.TIMES}
        
    def run(self):
        self.model.optimize()
        self._save_data()

    def calculate_results(self):
        if self.type == 'nodal':
            self.results.profits_G = {g:sum((self.data.lambda_[t]['N' + str(self.node_G[g])] - self.C_G_offer[g])  * self.data.generator_dispatch_values[g,t] for t in self.TIMES) for g in self.GENERATORS}
            self.results.profits_W = {w:sum(self.data.lambda_[t]['N' + str(self.node_W[w])] * self.data.wind_dispatch_values[w,t] for t in self.TIMES) for w in self.WINDTURBINES}
            self.results.utilities = {d:sum((self.U_D[t][d] - self.data.lambda_[t]['N' + str(self.node_D[d])]) * self.data.consumption_values[d,t] for t in self.TIMES) for d in self.DEMANDS}
        elif self.type == 'zonal':
            self.results.profits_G = {g:sum((self.data.lambda_[t][self.map_nz['N'+str(self.node_G[g])]] - self.C_G_offer[g])  * self.data.generator_dispatch_values[g,t] for t in self.TIMES) for g in self.GENERATORS}
            self.results.profits_W = {w:sum(self.data.lambda_[t][z] * self.data.wind_dispatch_values[w,t] for t in self.TIMES for z in self.ZONES) for w in self.WINDTURBINES}
            self.results.utilities = {d:sum((self.U_D[t][d] - self.data.lambda_[t][z]) * self.data.consumption_values[d,t] for t in self.TIMES for z in self.ZONES) for d in self.DEMANDS}

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Social welfare: " + str(self.data.objective_value))
        print()
        print("Profit of suppliers: ")
        print("Generators:")
        print(self.results.profits_G)
        print("Wind turbines:")
        print(self.results.profits_W)
        print()
        print("Utility of demands: ")
        print(self.results.utilities)
    
    def plot_prices(self):
        
        # Extract the time steps
        times = list(self.data.lambda_.keys())
        
        if self.type == 'zonal':
            # Extract zones
            zones = list(self.data.lambda_[times[0]].keys())
            
            # Define a list of colors and line styles
            colors = ['green', 'blue', 'orange']
            linestyle = ['-', '-', '--']

            # Plot the three zones
            for i, zone in enumerate(zones):
                lambda_values = [self.data.lambda_[t][zone] for t in times]
                plt.plot(times, lambda_values, drawstyle='steps', label=zone, color=colors[i], linestyle=linestyle[i], linewidth=3)

            # Add labels and legend
            plt.ylabel('Price [$/MWh]')
            plt.xlabel('Time')
            plt.legend() 
            plt.tight_layout()      
            plt.show()
        
        elif self.type == 'nodal':
            # Extract nodes
            nodes = list(self.data.lambda_[times[0]].keys())
            
            # Define a list of colors and line styles
            colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
            linestyles = ['-', '--']

            # Plot the nodal time series as stairs plots with unique colors and line styles
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            # Plot the first 12 graphs in the upper plot
            for i, node in enumerate(nodes[:12]):
                lambda_values = [self.data.lambda_[t][node] for t in times]
                color = colors[i % len(colors)]
                linestyle = linestyles[i//6 % len(linestyles)]
                ax1.plot(times, lambda_values, drawstyle='steps', label=node, color=color, linestyle=linestyle)

            # Plot the remaining graphs in the lower plot
            for i, node in enumerate(nodes[12:]):
                lambda_values = [self.data.lambda_[t][node] for t in times]
                color = colors[i % len(colors)]
                linestyle = linestyles[i//6 % len(linestyles)]
                ax2.plot(times, lambda_values, drawstyle='steps', label=node, color=color, linestyle=linestyle)

            # Add labels and legend
            ax1.set_ylabel('Price [$/MWh]', fontsize=16)
            ax2.set_xlabel('Time', fontsize=16)
            ax2.set_ylabel('Price [$/MWh]', fontsize=16)
            ax1.legend(loc = 'upper left', fontsize=12)
            ax2.legend(loc = 'upper left', fontsize=12)

            # Show the plot
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
            plt.show()

ec = NodalMarketClearing('nodal', True, True, True)
ec.variables.theta