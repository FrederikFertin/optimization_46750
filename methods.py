import gurobipy as gb

class CommonMethods:
    def _add_balance_constraints(self):
        """ Add balance constraints to the model - One for each hour """
        balance_constraints = {}

        for t in self.TIMES: # Add one balance constraint for each hour
            # Contribution of generators
            generator_expr = gb.quicksum(self.variables.generator_dispatch[g,t] for g in self.GENERATORS)

            # Contribution of demands
            demand_expr = gb.quicksum(self.variables.consumption[d,t] for d in self.DEMANDS)

            # Contribution of wind farms
            wind_expr = gb.quicksum(self.variables.wind_turbines[w,t]
                            for w in self.WINDTURBINES)
            if self.H2:
                # Contribution of wind farms with hydrogen production
                wind_expr = gb.quicksum(self.variables.wind_turbines[w,t] - self.variables.hydrogen[w,t]
                            for w in self.WINDTURBINES)
            
            # Contribution of batteries
            batt_expr = 0
            if self.battery:
                batt_expr = gb.quicksum(self.variables.battery_ch[b,t] - self.variables.battery_dis[b,t] 
                            for b in self.BATTERIES)

            # Central balance constraint
            balance_constraints[t] = self.model.addLConstr(
                                        demand_expr - generator_expr - wind_expr + batt_expr,
                                        gb.GRB.EQUAL,
                                        0, name='Balance equation')
        return balance_constraints
