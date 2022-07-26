import numpy as np
import scipy as sp

with open('simple-random.par.tab','w') as f:
    f.write('# Particle generated to test statisitics (Uniform in space; Standard Gaussian in velocity)\n')
    for i in range(256**3):
        x,y,z = np.random.uniform(low=-0.5,high=0.5,size=3)
        vx,vy,vz = np.random.normal(size=3)
        f.write(f'{i+1}  {x:.17f}  {y:.17f}  {z:.17f}  {vx:.17f}  {vy:.17f}  {vz:.17f}\n')

