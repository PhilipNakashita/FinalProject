from __future__ import division
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sympy
from sympy.simplify.simplify import simplify
from sympy.core.function import diff
from scipy.signal import cont2discrete
from ReferenceTrajectory import generateTrajectory

def cftoc(N,Q,R,xref, x0,xL,xU,uL,uU,Af=np.nan,bf=np.nan,stateNoise=np.array([0.0, 0.0, 0.0])):
    # Initialize values for LQR MPC problem
    numStates = 3
    numInputs = 2

    model = pyo.ConcreteModel()
    model.N = N
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


    def update_cost_function(model):
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

    model.cost = pyo.Objective(rule = update_cost_function, sense = pyo.minimize)


    model.dynamics_contraints = pyo.ConstraintList()
    for t in range(model.N - 1):
            Ts = 5
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
            T_Fluid = model.x[0, t]
            T_Reservoir = model.x[1, t]
            T_HEXplate = model.x[2, t]
           
            # Control Inputs
            V_TEC = model.u[0, t]                                              # Voltage Applied to TEC module
            q_HEX = model.u[1, t]*(T_HEXplate - T_Ambient)/0.01  # Liquid to Ambient Heat Exchanger TODO figure out MAX!!
           
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
           
            # Euler Discretization and Caluclation of Next State
            model.dynamics_contraints.add(expr = model.x[0, t+1] == Ts*Tdot_Fluid + T_Fluid + stateNoise[0])
            model.dynamics_contraints.add(expr = model.x[1, t+1] == Ts*Tdot_Reservoir + T_Reservoir + stateNoise[1])
            model.dynamics_contraints.add(expr = model.x[2, t+1] == Ts*Tdot_HEXplate + T_HEXplate+ stateNoise[2])

    # Initial value constraints
    model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
    model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
    model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])

    # Define state and input constraints
    model.constraint1= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[0,k]<= xU if k < model.N else pyo.Constraint.Skip)
    model.constraint2= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[0,k]>= xL if k < model.N else pyo.Constraint.Skip )
    model.constraint3= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[1,k]<= xU if k < model.N else pyo.Constraint.Skip)
    model.constraint4= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[1,k]>= xL if k < model.N else pyo.Constraint.Skip)
    model.constraint5= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[2,k]<= xU if k < model.N else pyo.Constraint.Skip)
    model.constraint6= pyo.Constraint(model.tidx, rule =lambda model, k: model.x[2,k]>= xL if k < model.N else pyo.Constraint.Skip)
    model.constraint9= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[0,k]<= uU[0] if k < model.N else pyo.Constraint.Skip)
    model.constraint10= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[0,k]>= uL[0] if k < model.N else pyo.Constraint.Skip)
    model.constraint11= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[1,k]<= uU[1] if k < model.N else pyo.Constraint.Skip)
    model.constraint12= pyo.Constraint(model.tidx, rule =lambda model, k: model.u[1,k]>= uL[1] if k < model.N else pyo.Constraint.Skip)

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

if __name__ == '__main__':
  timePoints = [0, 300, 900, 1200, 1800, 2100, 2600, 3000]
  TempPoints = [296, 296, 338, 338, 320, 320, 296, 296 ]
  Ts = 5
  t,Setpoint = generateTrajectory(timePoints,TempPoints,Ts)


  N = 10
  Q = np.array([[100, 0, 0], [0, 0, 0], [0, 0, 0]])
  R = np.identity(2)
  Xl = (273+5)
  Xu = (273+150)
  Ul = np.array([-12, 0])
  Uu = np.array([12, 1])
  x0 = np.array([273+23, 273+23, 273+23])

  x_actual = []
  u_actual = []
  x_cur = x0
  u_cur = [0,0]
  x_OL = np.zeros((3,N+1,len(t)))

  for i in range(len(t) - N):
    #print("Starting MPC")
    xref = np.array([Setpoint[i:i+N], Setpoint[i:i+N], Setpoint[i:i+N]])
    model, feas, xOpt, uOpt, JOpt = cftoc(N, Q, R, xref, x_cur, Xl, Xu, Ul, Uu)
    x_OL[:,:,i] = xOpt
    x_cur = xOpt[:,1]
    x_actual.append(xOpt[:,1])
    u_actual.append(uOpt[:,0])
    u_cur = uOpt[:,0]
    # line1 = plt.plot(t[i:i+N+1],xOpt[0,:], 'r--')
  print("Finished MPC")
  x_actual = np.array(x_actual)
  u_actual = np.array(u_actual)

  line2 = plt.plot(t,Setpoint, 'r--')
  line2 = plt.plot(t[0:len(t) - N],x_actual[:,0], 'b--')
  plt.show()
  plt.plot(u_actual)
  plt.show()
  
def sum_least_squares(reference_trajectory, modified_trajectory):

    # Convert trajectories to numpy arrays
    ref_array = np.array(reference_trajectory)
    mod_array = np.array(modified_trajectory)

    # Calculate the sum of squared differences
    squared_diff = (ref_array - mod_array)**2

    # Sum the squared differences
    sum_squared_diff = np.sum(squared_diff)

    return sum_squared_diff

from PIDControllerSim import x_pid
# Example usage
reference_trajectory = Setpoint[:591]
mpc_trajectory = x_actual[:,0]
pid_trajectory = x_pid[:591]

result = sum_least_squares(reference_trajectory, mpc_trajectory)
result2 = sum_least_squares(reference_trajectory, pid_trajectory)
print("Sum of Least Squares (MPC):", result)
print("Sum of Least Squares (PID):", result2)


line2 = plt.plot(t,Setpoint, 'b--',label = "Setpoint")
line2 = plt.plot(t,x_pid, 'r--', label = "PID (e\u00b2 = " + str(round(result2)) + ")")
line2 = plt.plot(t[0:len(t) - N],x_actual[:,0], 'g--', label = "MPC (e\u00b2 = " + str(round(result)) + ")" )

plt.legend()
plt.xlabel('time (in sec)')
plt.ylabel('Temperature (K)')
plt.title("MPC (N= 10) vs. PID controller")

plt.show()

