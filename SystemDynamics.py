import numpy as np

def thermoElectricTempControlModel(Ts, x, u):
  # Model Constant Values
  C_Fluid = 200.0
  C_Reservoir = 400.0
  C_HEXplate = 300.0

  R_Fluid_Ambient = 10.0
  R_Reservoir_Ambient = 3.0
  R_HEXplate_Ambient = 1.0
  R_Fluid_Reservoir = 0.2

  T_Ambient = 23.0 + 273

  R_TEC = 3.0              # Electrical Resistance of Thermoelectric device in Ohms
  alpha_TEC = 220 * 10**-4  # Seebeck Coefficient of TEC in V/Kelvin
  K_TEC = 1.5 * 10**-2      # Thermal Conductance of TEC between hot and cold side
  qmax_HEX = 80.0            # Maximum Cooling capability of Heat exchanger in Watts

  # State Variables
  T_Fluid = x[0]
  T_Reservoir = x[1]
  T_HEXplate = x[2]

  # Control Inputs
  V_TEC = u[0]                                              # Voltage Applied to TEC module
  q_HEX = min(u[1]*(T_HEXplate - T_Ambient)/0.01,qmax_HEX)  # Liquid to Ambient Heat Exchanger

  # Heat Flowrates
  q_Fluid_Ambient = (T_Fluid - T_Ambient)/R_Fluid_Ambient
  q_Reservoir_Ambient = (T_Reservoir - T_Ambient)/R_Reservoir_Ambient
  q_HEXplate_Ambient = (T_HEXplate - T_Ambient)/R_HEXplate_Ambient
  q_Fluid_Reservoir = (T_Reservoir - T_Fluid)/R_Fluid_Reservoir
  q_Reservoir_Fluid = -q_Fluid_Reservoir
  q_TEC_Reservoir =  alpha_TEC*(T_Reservoir)*V_TEC/R_TEC - 0.5*V_TEC**2/R_TEC + K_TEC*(T_Reservoir - T_HEXplate) # From Mathworks documentation of TEC modeling
  q_TEC_HEXplate = -alpha_TEC*(T_HEXplate)*V_TEC/R_TEC - 0.5*V_TEC**2/R_TEC + K_TEC*(T_HEXplate - T_Reservoir)

  # State Derivatives
  Tdot_Fluid = 1/C_Fluid*(-q_Fluid_Ambient + q_Fluid_Reservoir)
  Tdot_Reservoir = 1/C_Reservoir*(q_Fluid_Reservoir - q_Reservoir_Ambient - q_TEC_Reservoir)
  Tdot_HEXplate = 1/C_HEXplate*(q_HEXplate_Ambient - q_TEC_HEXplate - q_HEX)

  # Euler Discritization and Caluclation of Next State
  x_next = np.zeros((1,3))
  x_next[0,0] = Ts*Tdot_Fluid + T_Fluid
  x_next[0,1] = Ts*Tdot_Reservoir + T_Reservoir
  x_next[0,2] = Ts*Tdot_HEXplate + T_HEXplate
  x_next= np.array(x_next)

  return x_next