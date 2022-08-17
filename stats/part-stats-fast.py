"""
Code by Natascha 
"""
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import pandas as pds
import multiprocessing as multp
import os
import sys
from json import dump

def distance(x,y,boxl):
    #calculate the distance form point x=(x1,x2,x3) to point y=(y1,y2,y3)  
    sep = np.abs(x - y)
    dist = np.sqrt(np.sum((np.where(sep>boxl/2.,np.abs(sep-boxl),sep))**2,axis=-1))
    return dist

def velocitySq(vx,vy):
    #calculate the velocity difference magnitude sqared of point pair x,y  
    return np.sum((vx - vy)*(vx - vy),axis=-1)

def worker(subData,procnum, return_dict):
    """worker function"""
    global dataParticles
    global binEdge
    
    #build tree of all points global list
    pTree = cKDTree(dataParticles[['xps','yps','zps']].to_numpy(),boxsize=[1.0,1.0,1.0])    
    
    # allocate arrays for histograms
    vels    = np.zeros(binEdge.shape[0]-1)
    velsrad = np.zeros(binEdge.shape[0]-1)
    rdf     = np.zeros(binEdge.shape[0]-1)
    
    # loop over list of assigned particles
    for particle in subData.itertuples():
        
        #get postions and velocity of particle (particle is namedtuple!!!)
        partPos=np.array(particle[7:])
        partVel=np.array(particle[4:7])
        
        #get all points within maximum query radius from particle
        plist=pTree.query_ball_point(partPos,r=binEdge[-1])
        
        #get distances and velocity differences of particles paired with particle i
        posdiffs = distance(dataParticles.iloc[plist].loc[:,'xps':'zps'].to_numpy(),
                            partPos, boxl=np.array([1.0,1.0,1.0]))
        veldiffs = velocitySq(dataParticles.iloc[plist].loc[:,'vpx':'vpz'].to_numpy(),
                              partVel)
         
        # make distances and velocities at distances histograms
        rdf     += np.histogram(posdiffs,bins=binEdge)[0]
        vels    += stats.binned_statistic(posdiffs,veldiffs,statistic='sum',bins=binEdge)[0]
         
    return_dict[procnum] = np.stack((rdf,vels),axis=0)



file = ""
npart = -1
if len(sys.argv) > 2:
    file = sys.argv[1]
    npart = int(float(sys.argv[2]))
elif len(sys.argv) == 2:
    file = sys.argv[1]
else:
    raise SystemExit(f"No file name given. Usage: {sys.argv[0]} <file> ..arguments..")

if not (file.endswith('.par.tab')):
    raise SystemExit(f"File is not of type .par.tab.")

dataParticles = pds.read_csv(file, comment='#',header=None,delim_whitespace=True,index_col=0,
                     names=["id", "xp", "yp", "zp", "vpx", "vpy", "vpz"])

dataParticles['xps']=dataParticles['xp']+0.5
dataParticles['yps']=dataParticles['yp']+0.5
dataParticles['zps']=dataParticles['zp']+0.5

if (npart >=0 and npart<dataParticles.shape[0]):
    dataParticles = dataParticles.sample(n=npart)

bins = np.geomspace(1e-5,1e-0,num=61)
binEdge=bins[1:50:2]
binCenter=bins[2:50:2]


import os
nSublists= int(os.environ["SLURM_CPUS_PER_TASK"])
#nSublists=len(os.sched_getaffinity(0))

if __name__ == "__main__":
    
    #create n sublists with n = number of available processors
    sublists=np.array_split(dataParticles,nSublists)  
    
    #start manager to store return values
    manager = multp.Manager()
    return_dict = manager.dict()
    jobs = []
    
    #start multiprocessing
    for i in range(nSublists):
        p = multp.Process(target=worker, args=(sublists[i],i,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    # join results from all subprocesses
    totalCounts  =np.zeros(binEdge.shape[0]-1)
    totalVels    =np.zeros(binEdge.shape[0]-1)

    for i in range(nSublists):
        totalCounts  += return_dict[i][0]
        totalVels    += return_dict[i][1]
    
    #store results and bins in dictionary and dump to file
    mydict = { "binEdges":binEdge.tolist(), "binCenters":binCenter.tolist(),
               "partCounts":totalCounts.tolist(),"velocitySum":totalVels.tolist()
             }
    
    with open("RDFVelocityTest.json","w") as f:
        dump(mydict,f)
