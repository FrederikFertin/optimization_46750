#%%
import numpy as np
import pandas as pd
import os
#%%
class Network:
    wind = False

    # Reading data from Excel, requires openpyxl
    cwd = os.path.dirname(__file__)
    xls = pd.ExcelFile(cwd + '/data/grid_data.xlsx')
    
    ## Loading data from Excel sheets
    gen_tech = pd.read_excel(xls, 'gen_technical')
    gen_econ = pd.read_excel(xls, 'gen_cost')
    system_demand = pd.read_excel(xls, 'demand')
    line_info = pd.read_excel(xls, 'transmission_lines')
    load_info = pd.read_excel(xls, 'demand_nodes')
    wind_tech = pd.read_excel(xls, 'wind_technical')

    # Loading csv file of normalized wind profiles
    wind_profiles = pd.read_csv(cwd + '/data/wind_profiles.csv')

    # Load investment cost data
    investment_data = pd.read_excel(cwd + '/data/investment_costs.xlsx').iloc[:,:7].set_index('Metric')

    # Production profiles for wind and solar
    offshore_hourly_2019 = pd.read_csv(cwd + '/data/offshore_hourly_2019.csv', skiprows=3)['electricity'] # Wind production profiles for a 1 MW wind farm [pu]
    offshore_flux_dict = dict(zip(['T{0}'.format(int(key)+1) for key in offshore_hourly_2019.index], offshore_hourly_2019))
    onshore_hourly_2019 = pd.read_csv(cwd + '/data/onshore_hourly_2019.csv', skiprows=3)['electricity'] # Wind production profiles for a 1 MW wind farm [pu]
    onshore_flux_dict = dict(zip(['T{0}'.format(int(key)+1) for key in onshore_hourly_2019.index], onshore_hourly_2019))
    solar_hourly_2019 = pd.read_csv(cwd + '/data/pv_hourly_2019.csv', skiprows=3)['electricity'] # Solar production profiles for a 1 MW solar farm [pu]
    solar_flux_dict = dict(zip(['T{0}'.format(int(key)+1) for key in solar_hourly_2019.index], solar_hourly_2019))
    offshore_cf = sum(offshore_hourly_2019) / len(offshore_hourly_2019) # Offshore capacity factor
    onshore_cf = sum(onshore_hourly_2019) / len(onshore_hourly_2019) # Onshore capacity factor
    solar_cf = sum(solar_hourly_2019) / len(solar_hourly_2019) # Solar capacity factor
    cf = {'Solar': solar_cf, 'Onshore Wind': onshore_cf, 'Offshore Wind': offshore_cf, 'Nuclear': 0.9, 'Gas': 0.7}

    # Demand profiles
    demand_flux = pd.read_excel(cwd + '/data/load2023.xlsx').iloc[1:, :]
    demand_hourly = demand_flux[['Unnamed: 2']]
    demand_hourly.columns = ['Demand']
    demand_dict = dict(zip(['T{0}'.format(int(key)) for key in demand_hourly.index], demand_hourly['Demand']))

    ## Number of each type of unit/identity
    G = np.shape(gen_tech)[0] # Number of generators
    D = np.shape(load_info)[0] # Number of loads/demands
    T = np.shape(system_demand)[0] # Number of time periods/hours
    L = np.shape(line_info)[0] # Number of transmission lines
    W = np.shape(wind_tech)[0] # Number of wind farms
    # C = 5 # Number of technologies to invest in
    N = 24 # Number of nodes in network

    ## Lists of Generators etc.
    GENERATORS = ['G{0}'.format(t) for t in range(1, G+1)]
    DEMANDS = ['D{0}'.format(t) for t in range(1, D+1)]
    LINES = ['L{0}'.format(t) for t in range(1, L+1)]
    TIMES = ['T{0}'.format(t) for t in range(1, T+1)]
    
    if wind:
        WINDTURBINES = ['W{0}'.format(t) for t in range(1, W+1)]
    else:
        W = 0
        WINDTURBINES = []
    INVESTMENTS = list(investment_data.columns[0:5])    
    
    NODES = ['N{0}'.format(t) for t in range(1, N+1)]
    ZONES = ['Z1', 'Z2', 'Z3']

    # Zone to node mapping
    map_z = {'Z1': ['N17', 'N18', 'N21', 'N22'],
             'Z2': ['N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N19', 'N20', 'N23', 'N24'],
             'Z3': ['N{0}'.format(t) for t in range(1, 11)]}

    # Node to zone mapping
    map_nz = {n: z for z, ns in map_z.items() for n in ns}

    ## Investment Information
    investment_data = investment_data.transpose()
    CAPEX = dict(zip(INVESTMENTS, investment_data['CAPEX'][:-1])) # Capital expenditure [M€/MW]
    AF = dict(zip(INVESTMENTS, investment_data['AF'][:-1])) # Annualization factor [%]
    f_OPEX = dict(zip(INVESTMENTS, investment_data['f_OPEX'][:-1]/10**3)) # Fixed operational expenditure [M€/MW/year]
    v_OPEX = dict(zip(INVESTMENTS, investment_data['v_OPEX'][:-1]/10**6)) # Fixed operational expenditure [M€/MWh]
    LCOE = {}
    for key in INVESTMENTS:
        LCOE[key] = ((CAPEX[key] * AF[key] + f_OPEX[key])/(cf[key]*8760) + v_OPEX[key]) * 10**6
    node_I = dict(zip(INVESTMENTS, [['N1', 'N10', 'N16', 'N24']]*len(INVESTMENTS))) # Node placements of investments
    map_i = {}
    for number, node in enumerate(NODES):
        n = number + 1
        u_list = []
        for k, v in node_I.items():
            if node in v:
                u_list.append(k)
        map_i[node] = u_list

    ## Conventional Generator Information
    P_G_max = dict(zip(GENERATORS, gen_tech['P_max'])) # Max generation cap.
    P_G_min = dict(zip(GENERATORS, gen_tech['P_min'])) # Min generation cap.
    C_G_offer = dict(zip(GENERATORS, gen_econ['C'])) # Generator day-ahead offer price
    P_R_DW = dict(zip(GENERATORS, gen_tech['R_D'])) # Up-ramping of generator
    P_R_UP = dict(zip(GENERATORS, gen_tech['R_U'])) # Down-ramping of generator
    node_G_ = dict(zip(GENERATORS, gen_tech['Node'])) # Generator node placements
    P_R_PLUS = dict(zip(GENERATORS, gen_tech['R_plus'])) # Up reserve capacity
    P_R_MINUS = dict(zip(GENERATORS, gen_tech['R_minus'])) # Down reserve capacity
    C_U = dict(zip(GENERATORS, gen_econ['C_u'])) # Up reserve cost
    C_D = dict(zip(GENERATORS, gen_econ['C_d'])) # Down reserve cost
    EF = dict(zip(GENERATORS, gen_econ['EF [tCO2/MWh]'])) # Emission factor of generator [tCO2/MWh]
    
    ## Demand Information
    P_D_sum = dict(zip(TIMES, system_demand['System_demand'])) # Total hourly system demands [MWh]
    P_D = {} # Distribution of system demands
    for t, key in enumerate(TIMES):
        P_D[key] = dict(zip(DEMANDS, load_info['load_percent']/100 * system_demand['System_demand'][t]))
    
    U_D = dict(zip(DEMANDS, load_info['bid_price'])) # Demand bidding price <- set values in excel
    node_D_ = dict(zip(DEMANDS, load_info['Node'])) # Load node placements
    U_D_curt = 400 # cost of demand curtailment in BM [$/MWh]
    
    ## Wind Turbine Information
    C_W_offer = dict(zip(WINDTURBINES, W * [round(v_OPEX['Offshore Wind'] * 10**6,2)])) # Wind farm day-ahead offer price
    p_W_cap = 200 * wind # Wind farm capacities (MW)
    WT = ['V{0}'.format(v) for v in wind_tech['Profile']]
    chosen_wind_profiles = wind_profiles[WT] # 'Randomly' chosen production profiles for each wind farm
    P_W = {} # Wind production for each hour and each wind farm
    for t, key in enumerate(TIMES):
        P_W[key] = dict(zip(WINDTURBINES, chosen_wind_profiles.iloc[t,:] * p_W_cap))
    node_W_ = dict(zip(WINDTURBINES, wind_tech['Node'])) # Wind turbine node placements
    

    ## Transmission Line Information
    if wind:
        L_cap = dict(zip(LINES, line_info['Capacity_wind'])) # Capacity of transmission line [MVA]
    else:
        L_cap = dict(zip(LINES, line_info['Capacity_wind']))
    L_susceptance = dict(zip(LINES, 1/line_info['Reactance']))# [500]*L))# #  Susceptance of transmission line [pu.] 
    L_from = dict(zip(LINES, line_info['From'])) # Origin node of transmission line
    L_to = dict(zip(LINES, line_info['To'])) # Destination node of transmission line
    
    ## Inter-Zonal capacities
    # c_z1_z2 = L_cap['L25'] + L_cap['L27']
    # c_z2_z3 = L_cap['L7'] + L_cap['L14'] + L_cap['L15'] + L_cap['L16'] + L_cap['L17']

    # zone_cap = {'Z1': {'Z2': c_z1_z2},
    #             'Z2': {'Z1': c_z1_z2, 'Z3': c_z2_z3},
    #             'Z3': {'Z2': c_z2_z3}}
    # zonal = {'Z1': ['Z12'],
    #          'Z2': ['Z12', 'Z23'],
    #          'Z3': ['Z23']}
    # INTERCONNECTORS = ['Z12', 'Z23']
    # ic_cap = {'Z12': c_z1_z2,
    #           'Z23': c_z2_z3}


    def __init__(self):
        # Node to unit mappings
        self.map_g = self._map_units(self.node_G_) # Generators
        self.map_d = self._map_units(self.node_D_) # Demands
        self.map_w = self._map_units(self.node_W_) # Wind turbines
        # self.map_b = self._map_units(self.batt_node) # Batteries
        self.map_from = self._map_units(self.L_from) # Transmission lines
        self.map_to = self._map_units(self.L_to) # Transmission lines
        self._map_nodes() # Combination of the two above mappings
        self.node_W = {key : ["N{0}".format(value)] for key,value in self.node_W_.items()}
        self.node_G = {key : ["N{0}".format(value)] for key,value in self.node_G_.items()}
        self.node_D = {key : ["N{0}".format(value)] for key,value in self.node_D_.items()}
        self.node_L_from = dict(zip(self.LINES, ['N{0}'.format(n) for n in self.line_info['From']]))
        self.node_L_to = dict(zip(self.LINES, ['N{0}'.format(n) for n in self.line_info['To']]))

    def _map_units(self,node_list):
        mapping_units = {}
        for number, node in enumerate(self.NODES):
            n = number + 1
            u_list = []
            for k, v in node_list.items():
                if v == n:
                    u_list.append(k)
            mapping_units[node] = u_list
        return mapping_units
    
    def _map_nodes(self):
        self.map_n = {}
        for node_to, lines in self.map_to.items():
            self.map_n[node_to] = {}
            for line in lines:
                for node_from, lines_from in self.map_from.items():
                    if line in lines_from:
                        self.map_n[node_to][node_from] = line
        for node_from, lines in self.map_from.items():
            for line in lines:
                for node_to, lines_to in self.map_to.items():
                    if line in lines_to:
                        self.map_n[node_from][node_to] = line

if __name__ == '__main__':
    # Testing the Network class
    network = Network()
    print(network.LCOE)