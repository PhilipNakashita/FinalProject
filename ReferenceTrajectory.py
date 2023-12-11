import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def generateTrajectory(Times,Temperatures,Ts):
  Duration = Times[-1] - Times[0]
  numPoints = round(Duration/Ts)
  f = interpolate.interp1d(Times,Temperatures,'linear')
  tGrid = np.linspace(Times[0], Times[-1], numPoints+1)
  TGrid = f(tGrid)
  return tGrid, TGrid

## Example Trajectory Generation

if __name__ == '__main__':
  timePoints = [0, 300, 900, 1200, 1800, 2100]
  TempPoints = [23, 23, 65, 65, 23, 23]
  Ts = 5
  t,Setpoint = generateTrajectory(timePoints,TempPoints,Ts)

  plt.figure()
  plt.plot(t,Setpoint,'b*')
  plt.show()