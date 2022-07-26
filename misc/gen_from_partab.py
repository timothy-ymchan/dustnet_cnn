import argparse
import numpy as np
import scipy as sp

parser = argparse.ArgumentParser('Generate par.tab from existing  ones with randomized rhop or vp')

parser.add_argument('--in-partab',type=str,help='Input par.tab for generation',required=True)
parser.add_argument('--randomize',type=str,help='Property to randomize  (rhop or vp)',required=True)
parser.add_argument('--out-partab',type=str,help='Output par.tab file name',required=True)

args = parser.parse_args()

if args.randomize != 'rhop' and args.randomize != 'vp':
    print('Error: --randomize can only take "rhop" or "vp" as value')
    exit()

with open(args.in_partab,'r') as ifile, open(args.out_partab,'w') as ofile:

    ofile.write('# Particle generated to test statisitics (Uniform in space; Standard Gaussian in velocity)\n')
    for line in ifile:
        if line[0] == '#':
            continue
        pid,x,y,z,vx,vy,vz = [float(word) for word in line.split(' ') if len(word) != 0]
        #item =[float(word) for word in line.split('') if len(word) != 0]
        #print(item)
        if args.randomize == 'rhop':
            x,y,z = np.random.uniform(low=-0.5,high=0.5,size=3)
        elif args.randomize == 'vp':
            vx,vy,vz = np.random.normal(size=3)

        ofile.write(f'{pid}  {x:.17f}  {y:.17f}  {z:.17f}  {vx:.17f}  {vy:.17f}  {vz:.17f}\n')
