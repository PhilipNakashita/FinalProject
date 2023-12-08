import numpy as np
import matplotlib.pyplot as plt
from ThermalModel import thermoElectricTempControlModel
from ReferenceTrajectory import generateTrajectory


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
    self.integratedError = 0
    self.previousError = 0

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
    return np.clip(control_input,self.actuatorLL,self.actuatorUL)
  
  def computeDerivativeError(self,error):
    d_error = self.Kd * (error - self.previousError)/self.t_step
    self.error = error
    return d_error
  
  def computeIntegralError(self,error):
    self.integratedError += self.Ki * error *self.tstep
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

Ts = 5
Kp = 10
Ki = 0.1
Kd = 2

temperatureController = PID_Controller(5,Kp,Ki,Kd)
temperatureController.actuatorLL = -12
temperatureController.actuatorUL = 12

## Generate Reference Trajectory
timePoints = [0, 300, 900, 1200, 1800, 2100]
TempPoints = [23, 23, 65, 65, 23, 23]
t,Setpoint = generateTrajectory(timePoints,TempPoints,Ts)

## Simulate Thermal Control System
x0 = np.array([23, 23, 23])
x = x0
u = []
for k in range(t):
  u[k,0] = temperatureController.computeControlInput(Setpoint - x[k,0])
  u[k,1] = 0
  x_next = thermoElectricTempControlModel(Ts,x[k,:],u[k,:])
  x = np.append(x,x_next,axis=0)

plotResults(t,x)