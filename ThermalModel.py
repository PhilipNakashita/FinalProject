from __future__ import division
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class MPC_Controller():
  def __init__(self):
    pass


def thermoElectricTempControlModel(Ts, x, u):
  # Model Constant Values
  C_Fluid = 800
  C_Reservoir = 1000
  C_HEXplate = 600

  R_Fluid_Ambient = 10
  R_Reservoir_Ambient = 3
  R_HEXplate_Ambient = 1
  R_Fluid_Reservoir = 0.5

  T_Ambient = 23

  R_TEC = 0.02              # Electrical Resistance of Thermoelectric device in Ohms
  alpha_TEC = 220 * 10**-6  # Seebeck Coefficient of TEC in V/Kelvin
  K_TEC = 1.5 * 10**-3      # Thermal Conductance of TEC between hot and cold side
  qmax_HEX = 80             # Maximum Cooling capability of Heat exchanger in Watts

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
  q_TEC_Reservoir =  alpha_TEC*T_Reservoir*V_TEC/R_TEC - 0.5*V_TEC**2/R_TEC + K_TEC*(T_Reservoir - T_HEXplate) # From Mathworks documentation of TEC modeling
  q_TEC_HEXplate = -alpha_TEC*T_HEXplate*V_TEC/R_TEC - 0.5*V_TEC**2/R_TEC + K_TEC*(T_HEXplate - T_Reservoir)

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

def plotResults(t,x):
  fig = plt.figure(figsize=(9,5));
  plt.subplot(1,3,1)
  plt.plot(t,x[:,0])
  plt.subplot(1,3,2)
  plt.plot(t,x[:,1])
  plt.subplot(1,3,3)
  plt.plot(t,x[:,2])
  plt.show()

N = 50
Ts = 5

u = np.zeros((N,2))
for k in range(N):
  if k < 150:
    input = np.array([0,1])
  if k < 100:
    input = np.array([-5,1])
  if k < 50:
    input = np.array([-5,0])
  u[k,:] = input

x0 = np.ones((1,3))*23
x = x0
t = np.linspace(0,N*Ts,N+1)/60


for k in range(N):
  x_next = thermoElectricTempControlModel(Ts,x[k,:],u[k,:])
  x = np.append(x,x_next,axis=0)

#I will implement an MPC controller after thursday. Shouldn't take too long.

import sympy
from sympy.simplify.simplify import simplify
from sympy.core.function import diff

# Symbolic Variables
C_Fluid = sympy.Symbol("C_Fluid")
C_Reservoir = sympy.Symbol("C_Reservoir")
C_HEXplate = sympy.Symbol("C_HEXplate")
R_Fluid_Ambient = sympy.Symbol("R_Fluid_Ambient")
R_Reservoir_Ambient = sympy.Symbol("R_Reservoir_Ambient")
R_HEXplate_Ambient = sympy.Symbol("R_HEXplate_Ambient")
R_Fluid_Reservoir = sympy.Symbol("R_Fluid_Reservoir")
T_Ambient = sympy.Symbol("T_Ambient")
R_TEC = sympy.Symbol("R_TEC")
alpha_TEC = sympy.Symbol("alpha_TEC")
K_TEC = sympy.Symbol("K_TEC")
qmax_HEX = sympy.Symbol("qmax_HEX")

# Define generalized coordinates as functions of time
# and specify their derivatives for prettier computation
t = sympy.Symbol("t")
T_Fluid_func = sympy.Function("T_Fluid")
T_Reservoir_func = sympy.Function("T_Reservoir")
T_HEXplate_func = sympy.Function("T_HEXplate")
V_TEC_func = sympy.Function("V_TEC")
q_HEX_func = sympy.Function("q_HEX")

Tdot_Fluid_func = sympy.Function("Tdot_Fluid")
Tdot_Reservoir_func = sympy.Function("Tdot_Reservoir")
Tdot_HEXplate_func = sympy.Function("Tdot_HEXplate")

T_Fluid_func.fdiff = lambda self, argindex=1: T_Fluid_func(self.args[argindex-1])
T_Reservoir_func.fdiff = lambda self, argindex=1: T_Reservoir_func(self.args[argindex-1])
T_HEXplate_func.fdiff = lambda self, argindex=1: T_HEXplate_func(self.args[argindex-1])
V_TEC_func.fdiff = lambda self, argindex=1: V_TEC_func(self.args[argindex-1])
q_HEX_func.fdiff = lambda self, argindex=1: q_HEX_func(self.args[argindex-1])
Tdot_Fluid_func.fdiff = lambda self, argindex=1: Tdot_Fluid_func(self.args[argindex-1])
Tdot_Reservoir_func.fdiff = lambda self, argindex=1: Tdot_Reservoir_func(self.args[argindex-1])
Tdot_HEXplate_func.fdiff = lambda self, argindex=1: Tdot_HEXplate_func(self.args[argindex-1])

T_Fluid = T_Fluid_func(t)
T_Reservoir= T_Reservoir_func(t)
T_HEXplate = T_HEXplate_func(t)
V_TEC = V_TEC_func(t)
q_HEX = q_HEX_func(t)

# First derivatives
Tdot_Fluid = sympy.diff(T_Fluid, t)
Tdot_Reservoir = sympy.diff(T_Reservoir, t)
Tdot_HEXplate = sympy.diff(T_HEXplate, t)

q_Fluid_Ambient = (T_Fluid - T_Ambient)/R_Fluid_Ambient
q_Reservoir_Ambient = (T_Reservoir - T_Ambient)/R_Reservoir_Ambient
q_HEXplate_Ambient = (T_HEXplate - T_Ambient)/R_HEXplate_Ambient
q_Fluid_Reservoir = (T_Reservoir - T_Fluid)/R_Fluid_Reservoir
q_Reservoir_Fluid = -q_Fluid_Reservoir
q_TEC_Reservoir =  alpha_TEC*T_Reservoir*V_TEC/R_TEC - 0.5*V_TEC**2/R_TEC + K_TEC*(T_Reservoir - T_HEXplate) # From Mathworks documentation of TEC modeling
q_TEC_HEXplate = -alpha_TEC*T_HEXplate*V_TEC/R_TEC - 0.5*V_TEC**2/R_TEC + K_TEC*(T_HEXplate - T_Reservoir)

Tdot_Fluid = -q_Fluid_Reservoir/C_Fluid
Tdot_Reservoir =  (q_Fluid_Reservoir - q_Reservoir_Ambient - q_TEC_Reservoir)/C_Reservoir
Tdot_HEXplate = (-q_HEXplate_Ambient-q_HEX-q_TEC_HEXplate)/C_HEXplate

state_outputs = sympy.Matrix([[Tdot_Fluid],[Tdot_Reservoir],[Tdot_HEXplate]])
#state_outputs.jacobian([T_Fluid, T_Reservoir, T_HEXplate])
state_outputs.jacobian([V_TEC, q_HEX])

from scipy.signal import cont2discrete

B = np.array([[0,0],
              [-(((alpha_TEC*T_Reservoir)/R_TEC)+(V_TEC/R_TEC))/C_Reservoir,0],
              [(((alpha_TEC*T_HEXplate)/R_TEC)+(V_TEC/R_TEC))/C_HEXplate,-1/C_HEXplate]])
C = np.eye(3)
D = np.array([[0,0],[0,0],[0,0]])

Ad, Bd, Cd, Dd, dt = cont2discrete((A,B,C,D), dt, method='euler')

def equality_const_rule(self, model, i, t):
    return model.x[i, t+1] - (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX)
           + sum(model.B[i, j] * model.u[j, t] for j in model.uIDX))

model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=equality_const_rule)

model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])

def update_LD(self, x, u):
   # Model Constant Values
  C_Fluid = 800
  C_Reservoir = 1000
  C_HEXplate = 600

  R_Fluid_Ambient = 10
  R_Reservoir_Ambient = 3
  R_HEXplate_Ambient = 1
  R_Fluid_Reservoir = 0.5

  T_Ambient = 23

  R_TEC = 0.02              # Electrical Resistance of Thermoelectric device in Ohms
  alpha_TEC = 220 * 10**-6  # Seebeck Coefficient of TEC in V/Kelvin
  K_TEC = 1.5 * 10**-3      # Thermal Conductance of TEC between hot and cold side
  qmax_HEX = 80  

  # State Variables
  T_Fluid = x[0]
  T_Reservoir = x[1]
  T_HEXplate = x[2]

  # Control Inputs
  V_TEC = u[0]                                              # Voltage Applied to TEC module
  q_HEX = min(u[1]*(T_HEXplate - T_Ambient)/0.01,qmax_HEX)  # Liquid to Ambient Heat Exchanger

  A = np.array(A = np.array([[(1/(C_Fluid*R_Fluid_Reservoir)),-(1/(C_Fluid*R_Fluid_Reservoir)),0],
              [-(1/(C_Fluid*R_Fluid_Reservoir)),(-K_TEC-((alpha_TEC*V_TEC)/R_TEC)-(1/R_Reservoir_Ambient)+(1/R_Fluid_Reservoir))/(C_Reservoir) ,K_TEC/C_Reservoir],
              [0, K_TEC/C_HEXplate, ()/(C_HEXplate)]]))
  B = np.array([[0,0],
              [-(((alpha_TEC*T_Reservoir)/R_TEC)+(V_TEC/R_TEC))/C_Reservoir,0],
              [(((alpha_TEC*T_HEXplate)/R_TEC)+(V_TEC/R_TEC))/C_HEXplate,-1/C_HEXplate]])
  
return A, B 


