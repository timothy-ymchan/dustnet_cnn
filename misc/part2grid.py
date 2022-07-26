import numpy as np
import argparse
import pandas as pds
import datetime

parser = argparse.ArgumentParser(description='Converting par.tab files to .npy')
parser.add_argument('--par-tab',type=str,required=True,help='Path to par.tab file')
parser.add_argument('--rhop-out',type=str,required=True,help='Output file name of rhop')
parser.add_argument('--vp-out',type=str,required=True,help='Ouptut file name of vp')
parser.add_argument('--timestamp',action='store_true',help='Add timestamp to output files')

args = parser.parse_args()

with open(args.par_tab, 'r') as file:
    dataParticles = pds.read_csv(file, comment='#',header=None,delim_whitespace=True,index_col=0,
                     names=["id", "xp", "yp", "zp", "vpx", "vpy", "vpz"])

    # Center the particles
    dataParticles['xp'] = dataParticles['xp'] + 0.5
    dataParticles['yp'] = dataParticles['yp'] + 0.5
    dataParticles['zp'] = dataParticles['zp'] + 0.5

    # Create empty bins
    rhop = np.zeros(shape=(1,256,256,256)) 
    vp   = np.zeros(shape=(3,256,256,256))

    # Loop over frame
    to_bin = lambda x: np.floor(256*x)
    dataParticles['ip'] = to_bin(dataParticles.xp).astype('int')
    dataParticles['jp'] = to_bin(dataParticles.yp).astype('int')
    dataParticles['kp'] = to_bin(dataParticles.zp).astype('int')

    grouped = dataParticles.groupby(['ip','jp','kp'])
    
    for idc, group in grouped:
        ip,jp,kp = idc
        #print(ip,jp,kp)
        #print(group.dtypes)
        rhop[0,ip,jp,kp] += group['vpx'].count() # Just count the number of particles along any col
        vp[0,ip,jp,kp]   += group['vpx'].sum()
        vp[1,ip,jp,kp]   += group['vpy'].sum()
        vp[2,ip,jp,kp]   += group['vpz'].sum()

    vp = np.divide(vp,rhop,out=np.zeros_like(vp), where=(rhop!=0)) # Average over grids

    rhop_name = args.rhop_out
    vp_name   = args.vp_out

    if args.timestamp:
        ct = datetime.datetime.now()
        ts = int(ct.timestamp())
        timestamp = f'{ts}'
        rhop_name = rhop_name + '-' + timestamp
        vp_name   = vp_name   + '-' + timestamp
    rhop_name = rhop_name + '.npy'
    vp_name   = vp_name   + '.npy'

    np.save(rhop_name,rhop)
    np.save(vp_name,vp)
