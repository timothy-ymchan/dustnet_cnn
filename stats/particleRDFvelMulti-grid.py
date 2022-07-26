import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import multiprocessing as multp
import os 
import sys 
import json
import argparse
import datetime

# Global variables
bins = np.geomspace(2/256,1e-1,num=41)
binEdge=bins[1:40:2]
binCenter=bins[2:40:2]

nSublists = None

rhop = vp = nsamples = boxl = x1f = x2f = x3f = x1v = x2v = x3v = XVGrid = XVPts = samplesId = ncpu = None

args = None

def init():
    
    global rhop , vp , nsamples , boxl , x1f , x2f , x3f , x1v , x2v , x3v , XVGrid , XVPts ,nSublists, samplesId, ncpu, args
    
    args = get_args()
    
    rhop = np.load(args.rhop)
    vp = np.load(args.vp)
    assert rhop.shape[1:] == vp.shape[1:], 'vp and rhop should have same spatial dimension'
    
    nsamples = args.nsamples if not(args.nsamples is None) else rhop.flatten().shape[0]
    
    samplesId = np.random.choice(np.arange(rhop.flatten().shape[0]),nsamples, replace=False)
    samplesId = samplesId[:256*256*100] # comment out later
    
    
    boxl = np.array(args.boxl)
    
    x1f, x2f, x3f = [np.linspace(0,boxl[i],num=rhop.shape[i+1]+1) for i in range(3)]
    x1v, x2v, x3v = 0.5*(x1f[1:] + x1f[:-1]),0.5*(x2f[1:] + x2f[:-1]),0.5*(x3f[1:] + x3f[:-1])
    
    XVGrid = make_grid(x1v,x2v,x3v)
    XVPts  = make_pts(XVGrid)
    
    rhop = make_pts(rhop)
    vp = make_pts(vp)
    
    try:
        ncpu = int(os.environ['SLURM_CPUS_PER_TASK'])
    except:
        print('Unable to find environment variable "SLURM_CPUS_PER_TASK"! Automatically setting ncpu = 1')
        ncpu = 1
        
    
    #print(nsamples,npyRhop,npyVp)
    
def worker(subData, procnum,return_dict):
    global rhop , vp , nsamples , boxl , x1f , x2f , x3f , x1v , x2v , x3v , XVGrid , XVPts ,nSublists, samplesId
    global bins,binEdge,binCenter
    
    print('Task:', procnum)
    print('Subdata: ',subData)
    print('Count: ',len(subData))
    
    # Construct the tree
    pTree = cKDTree(XVPts,boxsize=boxl)
    
    # Compute statistics for each particle in sample
    vels    = np.zeros(binEdge.shape[0]-1) # Relative velocity
    velsrad = np.zeros(binEdge.shape[0]-1) # Relative radial velocity
    rdf     = np.zeros(binEdge.shape[0]-1) # RDF
    sumRhop = 0
    pCount = 0

    for p in subData:
        if pCount % 256 == 0:
            print(f'Handling {pCount}th Particle (PID:{p})')
        # Particle at Center
        pPos = XVPts[p]
        pRho = rhop[p]
        pVel = vp[p]
        
        # Particles inside a ball of radius binEdge[-1]
        qList = pTree.query_ball_point(pPos,r=binEdge[-1])
        print(len(qList))
        qPos = np.take(XVPts,qList,axis=0)
        qVel = np.take(vp,qList,axis=0)
        qRho = np.take(rhop,qList,axis=0).squeeze()
        
        # Pairwise distance and velocity difference
        rij_hat, rij = distance(qPos,pPos,boxl=boxl)
        v_diff = velocitySq(qVel,pVel)
        v_diff_rad = velocityRadial(qVel,pVel,rij_hat)
        
        
        # Radial binning to compute statistics
        avgNShells = stats.binned_statistic(rij,qRho,statistic='mean',bins=binEdge)[0]
        rhopShells = stats.binned_statistic(rij,qRho,statistic='sum',bins=binEdge)[0]
        momentumShells = stats.binned_statistic(rij,qRho*v_diff,statistic='sum',bins=binEdge)[0]
        weightedVelcShells = momentumShells/rhopShells

        print(binEdge)
        print(avgNShells)#,rhopShells,momentumShells)
        
        momentumRadialShells = stats.binned_statistic(rij,qRho*v_diff_rad,statistic='sum',bins=binEdge)[0]
        weightedVelcRadialShells = momentumRadialShells / rhopShells
        
        
        # Update
        if np.isfinite(rhopShells).all() and np.isfinite(avgNShells).all() and np.isfinite(weightedVelcShells).all() and np.isfinite(weightedVelcRadialShells).all():
            rdf += pRho*avgNShells
            vels += pRho* weightedVelcShells
            velsrad += pRho*weightedVelcRadialShells
            sumRhop += pRho

            pCount += 1

    #vels /= sumRhop # Can delay this division to the last part
    #velsrad /= sumRhop
    #print(vels/sumRhop)
    #print(velsrad/sumRhop)
    
    return_dict[procnum] = {'sumRhop':sumRhop, 'stats':np.stack((rdf,vels,velsrad),axis=0)}
    
    #subData = subData[:10]
    
    #xi_sample = np.take(XVPts,subData,axis=0)
    #vp_sample = np.take(vp,subData,axis=0)
    #rhop_sample = np.take(rhop,subData,axis=0)
    
    #print(xi_sample, vp_sample,rhop_sample)
    #print(xi_sample.shape, vp_sample.shape, rhop_sample.shape)

def distance(x,y,boxl):
    #calculate the distance form point x=(x1,x2,x3) to point y=(y1,y2,y3)  
    sep = x - y
    sep = np.where(sep>boxl/2.,sep-boxl,sep)
    sep = np.where(sep< -boxl/2.,sep+boxl,sep)
    
    dist = np.sqrt(np.sum(sep**2,axis=-1))
    
    #return x-y/|x-y| and |x-y|, returns zeros for entry x=y
    return np.divide(sep,dist[:,None],out=np.zeros_like(sep),where=sep!=0) , dist


def velocitySq(vx,vy):
    #calculate the velocity difference magnitude sqared of point pair x,y  
    return np.sum((vx - vy)*(vx - vy),axis=-1)

def velocityRadial(vx,vy,diff_uvec):
    #calculate the velocity difference magnitude sqared of point pair x,y  
    return np.absolute(np.sum((vx - vy)*diff_uvec,axis=-1))


def make_grid(*xi,indexing='xy'): # "xy" for k,j,i and "ij" for i,j,k
    if len(xi) == 2:
        return np.stack(np.meshgrid(*xi,indexing=indexing))
    grid = np.meshgrid(*xi,indexing='ij')
    if indexing == 'xy':
        return np.stack([g.reshape(g.shape[::-1]) for g in  grid])
    return np.stack(grid)

def make_pts(grid): # Reshape coordinate grid to coordinate points
    #print(grid.shape)
    grid = np.moveaxis(grid,0,-1) # (no_coord,n1,n2,n3) -> (n1,n2,n3,no_coord)
    #print(grid.shape)
    return grid.reshape(-1,grid.shape[-1])

def get_args():
    parser = argparse.ArgumentParser(description='particleRDFvelMulti on Grid')
    parser.add_argument('--nsamples', type=int,help='Number of sampling grids. Default is all grids')
    parser.add_argument('--rhop',type=str,required=True,help='Path to rhop (npy)')
    parser.add_argument('--vp',type=str,required=True,help='Path to vp (npy)')
    parser.add_argument('--boxl',type=list,help='Box size in simulation unit',default=[1.0,1.0,1.0])
    parser.add_argument('--outname',type=str,help='Output name',default=None)
    parser.add_argument('--timestamp',action='store_true')
            
    return parser.parse_args()


init()
if __name__ == "__main__":
    init()
    print(samplesId,ncpu)
    sublists = np.array_split(samplesId,ncpu) # Use all cpu to spread the tasks
    
    manager = multp.Manager()
    return_dict = manager.dict()
    jobs = []
    
    for i in range(ncpu):
        p = multp.Process(target=worker,args=(sublists[i],i,return_dict))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()
    
    sumRhops = sum([return_dict[i]['sumRhop'] for i in range(ncpu)])
    
    
    # join results from all subprocesses
    totalRDF  =np.zeros(binEdge.shape[0]-1)
    totalVels    =np.zeros(binEdge.shape[0]-1)
    totalVelsRad =np.zeros(binEdge.shape[0]-1)
    for i in range(ncpu):
        totalRDF     += return_dict[i]['stats'][0]
        totalVels    += return_dict[i]['stats'][1]
        totalVelsRad += return_dict[i]['stats'][2]
    
    totalRDF /= sumRhops
    totalVels /= sumRhops
    totalVelsRad /= sumRhops
    
    outDict = {'rhop-path':args.rhop,'vp-path':args.vp,'nsamples':nsamples,'binEdges':binEdge.tolist(),'binCenters':binCenter.tolist(),
               'rdf':totalRDF.tolist(),'velc_sum':totalVels.tolist(),'velc_rad_sum':totalVelsRad.tolist()}
    
    ct = datetime.datetime.now()
    
    if args.outname is None:
        outname = "RDFVelocityTest"
    else:
        outname = args.outname
    
    if args.timestamp:
        outname = f'{outname}-{ct.timestamp():.0f}'

    with open(f'{outname}.json','w') as f:
        json.dump(outDict,f)
               
    #print(sublists)
    #print(ncpu)
    #print(XVPts.shape,rhop.shape,vp.shape)
