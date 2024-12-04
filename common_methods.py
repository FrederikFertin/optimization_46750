import numpy as np

class CommonMethods:

    def _initialize_fluxes_demands(self):
        self.TIMES = self.chosen_hours
        if self.T < 24: # If typical days are used, set the number of hours to 24
            self.solar_flux_dict    = dict(zip(self.TIMES, np.ones(self.T)))
            self.onshore_flux_dict  = dict(zip(self.TIMES, np.ones(self.T)))
            self.offshore_flux_dict = dict(zip(self.TIMES, np.ones(self.T)))
        else:
            self.cf = {i: 1 for i in self.INVESTMENTS} # If typical days are used, set capacity factors to 1.
            self.P_D = {} # Distribution of system demands
            demand_gain = 1 # Increase system demand for all hours and nodes
            for t in self.TIMES:
                self.P_D[t] = dict(zip(self.DEMANDS, self.load_info['load_percent']/100 * self.demand_dict[t] * demand_gain))
        self.nuclear_flux_dict      = dict(zip(self.TIMES, np.ones(self.T)))
        self.gas_flux_dict          = dict(zip(self.TIMES, np.ones(self.T)))
        # Establish fluxes (primarily capping generation capacities of renewables)
        self.fluxes = {'Onshore Wind': self.onshore_flux_dict,
                        'Offshore Wind': self.offshore_flux_dict,
                        'Solar': self.solar_flux_dict,
                        'Nuclear': self.nuclear_flux_dict,
                        'Gas': self.gas_flux_dict}

    def _initialize_costs(self):        
        # Define generation costs
        self.PRODUCTION_UNITS = self.GENERATORS + self.WINDTURBINES + self.INVESTMENTS
        self.node_production = {**self.node_G, **self.node_I, **self.node_W}
        self.C_G_offer_modified = {g: round(self.C_G_offer[g] + (self.EF[g]*self.carbontax), 2) for g in self.GENERATORS} # Variable costs in €/MWh incl. carbon tax
        self.C_I_offer = {g: round(self.v_OPEX[g] * 10**6, 2) for g in self.INVESTMENTS} # Variable costs in €/MWh
        self.C_offer = {**self.C_G_offer_modified, **self.C_I_offer, **self.C_W_offer} # Indexed by PRODUCTION_UNITS

