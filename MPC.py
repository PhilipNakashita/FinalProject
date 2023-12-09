from ReferenceTrajectory import generateTrajectory, Setpoint
import numpy as np
import ThermalModel
import importlib
import pyomo.environ as pyo
import matplotlib.pyplot as plt

class MPC_Controller():
   
    def __init__(self,A,B,N,Q,R,Xl,Xu,Ul,Uu,x0,xref):
        self.model = pyo.ConcreteModel()

        self.model.A = A
        self.model.B = B
        self.model.N = N
        self.model.Q = Q
        self.model.R = R
        self.model.x0 = x0
        self.model.xref = xref
        self.model.Xl = Xl
        self.model.Xu = Xu
        self.model.Ul = Ul
        self.model.Uu = Uu
        self.init_variables()
       
        self.model.cost = pyo.Objective(rule = lambda model: self.objective_function(), sense = pyo.minimize)
        self.set_state_constraints()
        self.set_input_constraints()
        self.set_dynamics_constraints()
        self.solve()
        self.x_opt = np.asarray([[self.model.x[i,t]() for i in self.model.xidx] for t in self.model.tidx]).T
        self.u_opt = self.model.u
       

    def init_variables(self):
        self.model.num_states = np.size(A,0)
        self.model.num_inputs = np.size(B,1)
        self.model.tidx = pyo.Set(initialize = range(self.model.N + 1))
        self.model.xidx = pyo.Set(initialize = range(self.model.num_states))
        self.model.uidx = pyo.Set(initialize = range(self.model.num_inputs))
        self.model.x = pyo.Var(self.model.xidx, self.model.tidx)
        self.model.u = pyo.Var(self.model.uidx, self.model.tidx)

    def set_state_constraints(self):
        self.model.state_contraints = pyo.ConstraintList()
        for t in range(self.model.N):
            for i in range(self.model.num_states):
                self.model.state_contraints.add(expr = self.model.x[i, t] >= self.model.Xl[i])
                self.model.state_contraints.add(expr = self.model.x[i, t] <= self.model.Xu[i])
        for i in range(self.model.num_states):
            self.model.state_contraints.add(expr = self.model.x[i, 0] == self.model.x0[i])

    def set_input_constraints(self):
        self.model.input_contraints = pyo.ConstraintList()
        for t in range(self.model.N - 1):
            for i in range(self.model.num_inputs):
                self.model.input_contraints.add(expr = self.model.u[i, t] >= self.model.Ul[i])
                self.model.input_contraints.add(expr = self.model.u[i, t] <= self.model.Uu[i])

    def set_dynamics_constraints(self):
        self.model.dynamics_contraints = pyo.ConstraintList()
        for t in range(self.model.N - 1):
            for i in range(self.model.num_states):
                self.model.dynamics_contraints.add(expr =
                                                   (sum(self.model.A[i, j] * self.model.x[j, t] for j in self.model.xidx)
                                                    + sum(self.model.B[i, j] * self.model.u[j, t] for j in self.model.uidx)
                                                    == self.model.x[i, t+1]))

    def objective_function(self):
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

    def solve(self):
        solver = pyo.SolverFactory('ipopt')
        results = solver.solve(self.model)
