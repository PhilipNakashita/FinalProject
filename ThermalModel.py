from __future__ import division
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sympy
from sympy.simplify.simplify import simplify
from sympy.core.function import diff
from scipy.signal import cont2discrete

class PID_Controller():
  def __init__(self):
    # Initialize Default Gains
    self.Kp = 1
    self.Ki = 1
    self.Kd = 1

    pass

  def setProportionalGain(self,P):
    pass

  def setIntegralGain(self,I):
    pass

  def setDerivativeGain(self,D):
    pass

  def setIntegralWindupMode(self,mode):
    pass

  def computeControlInput(self):
    pass

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

class MPC_Controller():
  def __init__(self,A,B,N,Q,R,x0,Af=[],bf=[]):
    # Initialize values for LQR MPC problem
    self.numStates = np.size(A,0)
    self.numInputs = np.size(B,1)
    self.M = M
    xL = 5
    xU = 150
    u = np.array([-12,0],[12,1])
    uL = np.array([-12,0])
    uU = np.array([12,1]) 
    

    self.model = pyo.ConcreteModel()
    self.model.N = N
    self.model.A = A
    self.model.B = B
    self.model.P = P
    self.model.Q = Q
    self.model.R = R
    self.model.Af = Af
    self.model.bf = bf
    self.model.x0 = x0
    self.model.xref = []

    self.model.tidx = pyo.Set(initialize = range(self.model.N + 1))
    self.model.xidx = pyo.Set(initialize = range(self.numStates))
    self.model.uidx = pyo.Set(initialize = range(self.numInputs))

    self.model.x = pyo.Var(self.model.xidx, self.model.tidx)
    self.model.u = pyo.Var(self.model.uidx, self.model.tidx)
    pass

  def solve_cftoc(self,A,B,xref):
    self.model.xref = xref
    self.model.A = A
    self.model.B = B
    self.update_cost_function()
    self.set_state_constraints()

  def set_state_constraints(self):
    self.model.StateConstraint = pyo.ConstraintList()
    if np.any(self.model.Af) == True:
      for i in range(np.size(self.model.bf)):
        self.model.StateConstraint.add(expr = sum(self.model.Af[i,j]*self.model.x[j,N] for j in range(np.size(self.model.Af,1))) <= self.model.bf[i])
     else:
      for i in self.model.xidx:
        self.model.StateConstraint.add(expr = self.model.x[i,N] == self.model.bf[i])
    pass

  def set_input_constraints(self,B_f):
    pass

  def set_terminal_constraints(self,X_f):
    pass

  def set_state_cost_matrix(self,Q):
    try:
      self.model.Q = Q
      return True
    except:
      return False

  def set_input_cost_matrix(self,R):
    try:
      self.model.R = R
      return True
    except:
      return False

  def update_cost_function(self):
    stateCost = 0.0
    inputCost = 0.0
    for t in self.model.tidx:
      for i in self.model.xidx:
        for j in self.model.xidx:
          if t < self.model.N:
            stateCost += (self.model.x[i,t] - self.model.xref[i,t])*self.model.Q[i,j]*(self.model.x[j,t] - self.model.xref[j,t])
    for t in self.model.tidx:
      for i in self.model.uidx:
        for j in self.model.uidx:
          if t < self.model.N:
            inputCost += self.model.u[i,t]*self.model.R[i,j]*self.model.u[j,t]
    return stateCost + inputCost

    pass

  B = np.array([[0,0],
              [-(((alpha_TEC*T_Reservoir)/R_TEC)+(V_TEC/R_TEC))/C_Reservoir,0],
              [(((alpha_TEC*T_HEXplate)/R_TEC)+(V_TEC/R_TEC))/C_HEXplate,-1/C_HEXplate]])
  C = np.eye(3)
  D = np.array([[0,0],[0,0],[0,0]])

  Ad, Bd, Cd, Dd, dt = cont2discrete((A,B,C,D), dt, method='euler')

Q = np.array([[100,0,0],[0,1,0],[0,0,1]]) # expand to N number of steps
R = np.array([[1,0],[0,1]])
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

A = np.array([[(1/(C_Fluid*R_Fluid_Reservoir)),-(1/(C_Fluid*R_Fluid_Reservoir)),0],
              [-(1/(C_Fluid*R_Fluid_Reservoir)),(-K_TEC-((alpha_TEC*V_TEC)/R_TEC)-(1/R_Reservoir_Ambient)+(1/R_Fluid_Reservoir))/(C_Reservoir) ,K_TEC/C_Reservoir],
              [0, K_TEC/C_HEXplate, ()/(C_HEXplate)]])
def cftoc(A,B,N,Q,R,xref, x0,xL,xU,uL,uU,Af=np.nan,bf=np.nan):
    # Initialize values for LQR MPC problem
    numStates = np.size(A,0)
    numInputs = np.size(B,1)

    model = pyo.ConcreteModel()
    model.N = N
    model.A = A
    model.B = B
    model.P = P
    model.Q = Q
    model.R = R
    model.Af = Af
    model.bf = bf
    model.x0 = x0
    model.xref = xref

    model.tidx = pyo.Set(initialize = range(model.N + 1), ordered=True )
    model.xidx = pyo.Set(initialize = range(numStates), ordered=True )
    model.uidx = pyo.Set(initialize = range(numInputs), ordered=True )

    model.x = pyo.Var(model.xidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)


    def update_cost_function():
      stateCost = 0.0
      inputCost = 0.0
      for t in model.tidx:
        for i in model.xidx:
          for j in model.xidx:
            if t < model.N:
              stateCost += (model.x[i,t] - model.xref[i,t])*model.Q[i,j]*(model.x[j,t] - model.xref[j,t])
      for t in model.tidx:
        for i in model.uidx:
          for j in model.uidx:
            if t < model.N:
              inputCost += model.u[i,t]*model.R[i,j]*model.u[j,t]
      return stateCost + inputCost

    model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)

    # Terminal constraint
    def equality_const_rule(model, i, t):
      return model.x[i, t+1] - (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX)
        + sum(model.B[i, j] * model.u[j, t] for j in model.uIDX))

    model.equality_constraints = pyo.Constraint(model.xidx, model.tidx, rule=equality_const_rule)

    # Initial value constraints
    model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
    model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
    model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])


    # Define state and input constraints
    model.constraint1= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[0,k]<= xU if k<N else pyo.Constraint.Skip)
    model.constraint2= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[0,k]>= xL if k<N else pyo.Constraint.Skip)
    model.constraint3= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[1,k]<= xU if k<N else pyo.Constraint.Skip)
    model.constraint4= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[1,k]>= xL if k<N else pyo.Constraint.Skip)
    model.constraint5= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[2,k]<= xU if k<N else pyo.Constraint.Skip)
    model.constraint6= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[2,k]>= xL if k<N else pyo.Constraint.Skip)
    model.constraint7= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[3,k]<= xU if k<N else pyo.Constraint.Skip)
    model.constraint8 = pyo.Constraint(model.tidx, rule =lambda model, k: model.x[3,k]>= xL if k<N else pyo.Constraint.Skip)
    model.constraint9= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[0,k]<= uU[0]if k<N else pyo.Constraint.Skip)
    model.constraint10= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[0,k]>= uL[0] if k<N else pyo.Constraint.Skip)
    model.constraint11= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[1,k]<= uU[1] if k<N else pyo.Constraint.Skip)
    model.constraint12= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[1,k]>= uL[1] if k<N else pyo.Constraint.Skip)

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    if str(results.solver.termination_condition) == "optimal":
        feas = True
    else:
        feas = False

    xOpt = np.asarray([[model.x[i,t]() for i in model.xidx] for t in model.tidx]).T
    uOpt = np.asarray([model.u[:,t]() for t in model.tidx]).T

    JOpt = model.cost()

    return [model, feas, xOpt, uOpt, JOpt]

print('JOpt=', JOpt)

fig = plt.figure(figsize=(9, 6))
plt.plot(xOpt.T)
plt.ylabel('x')
fig = plt.figure(figsize=(9, 6))
plt.plot(uOpt.T)
plt.ylabel('u')
plt.show()
