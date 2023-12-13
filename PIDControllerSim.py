import numpy as np
import matplotlib.pyplot as plt
from SystemDynamics import thermoElectricTempControlModel, simLinearSystem, linearizeModel
from ReferenceTrajectory import generateTrajectory
from scipy.linalg import eig

class PID_Controller():
  def __init__(self,t_step,Kp,Ki,Kd):
    # Initialize Timestep
    self.tstep = t_step

    # Initialize Gains
    self.Kp = Kp
    self.Ki = Ki
    self.Kd = Kd

    '''
    Defines integral windup mode:
    -   True: integral will not accumulate when actuators are saturated
    -   False: integral will continue to accumulate
    '''
    self.windupClamp = False

    # initializes integrated error and previous error for Ki and Kd
    self.integratedError = 0.0
    self.previousError = 0.0

    # Initialize Actuator Limits
    self.actuatorLL = np.nan
    self.actuatorUL = np.nan

    pass

  def setProportionalGain(self,P):
    self.Kp = P
    pass

  def setIntegralGain(self,I):
    self.Ki = I
    pass

  def setDerivativeGain(self,D):
    self.Kd = D
    pass

  def setIntegralWindupMode(self,mode):
    self.windupClamp = mode
    pass

  def computeControlInput(self,error):
    control_input = self.Kp * error + self.computeIntegralError(error) + self.computeDerivativeError(error)
    if control_input > self.actuatorUL:
      control_input = self.actuatorUL

    if control_input < self.actuatorLL:
      control_input = self.actuatorLL

    return control_input
  
  def computeDerivativeError(self,error):
    d_error = self.Kd * (error - self.previousError)/self.tstep
    self.error = error
    return d_error
  
  def computeIntegralError(self,error):
    self.integratedError += self.Ki * error * self.tstep
    return self.integratedError

  def setActuatorLimits(self,lowerLimit,upperLimit):
    self.actuatorLL = lowerLimit
    self.actuatorUL = upperLimit

def plotResults(t,x):
  fig = plt.figure(figsize=(9,5))
  plt.subplot(1,3,1)
  plt.plot(t,x[:,0])
  plt.subplot(1,3,2)
  plt.plot(t,x[:,1])
  plt.subplot(1,3,3)
  plt.plot(t,x[:,2])
  plt.show()

Ts = 5.0
Kp = 25.0
Ki = 0.1
Kd = 0.4

temperatureController = PID_Controller(Ts,Kp,Ki,Kd)
temperatureController.setActuatorLimits(-12,12)

## Generate Reference Trajectory
timePoints = [0, 300, 900, 1200, 1800, 2100, 2600, 3000]
TempPoints = [296, 296, 338, 338, 320, 320, 296, 296 ]
t,Setpoint = generateTrajectory(timePoints,TempPoints,Ts)

## Simulate Thermal Control System
x0 = np.ones((1,3))*(23 + 273)
x = x0
u = np.zeros((len(t),2))
error = np.zeros((1,1))

# Nonlinear Sym
for k in range(len(t)):
  error = Setpoint[k] - x[k,0]
  u[k,0] = -temperatureController.computeControlInput(error)
  if u[k,0] > 0:
    u[k,1] = u[k,0]/12.0
  else:
    u[k,1] = 0
  print(u[k,:])
  x_next = thermoElectricTempControlModel(Ts,x[k,:],u[k,:])
  x = np.append(x,x_next,axis=0)

# Linear Sim
x_lin = x0
u_lin = np.zeros((len(t),2))
Ad, Bd = linearizeModel(Ts, x_lin[0,:],u_lin[0,:])
print(eig(Ad))
print(f"Ad = {Ad}")
print(f"Bd = {Bd}")
for k in range(len(t)):
  u_lin[k,0] = -temperatureController.computeControlInput(Setpoint[k] - x_lin[k,0])
  u_lin[k,1] = 0
  
  x_lin_next = simLinearSystem(Ad, Bd, x_lin[k,:],u_lin[k,:])
  x_lin = np.append(x_lin,x_lin_next,axis=0)

plt.figure()
plt.plot(t, x[0:len(t),0], 'b--')
plt.plot(t,Setpoint,'r--')
plt.show()