import numpy as np
import time
import multiprocessing as multip
from scipy import stats
import glob
import athena_read
import h5py
import matplotlib.pyplot as plt

def main():
    normal = read_athdf('../athenaMLdata/data/*103*')
    rs = np.geomspace(1/128,1,10)
    nproc=5
    for i,r in enumerate(rs):
        print('Job ',i+1)
        print('Number of processors: ',nproc)
        st = time.time()
        s3_val = S3longitudinal_Sampling_Multiproc(data=normal[0],r=r,nsamples=1000000,nproc=nproc) # 1000000 samples run
        write_line('./s3long-out/s3-normal-sampling-1000000-50-pts.txt',f"{r},{s3_val}") 
        et = time.time()
        print(f'Processing time: {et-st} sec')
        if i % 5 == 0:
            print('Iteration: ',i+1,'r=',r)



def read_athdf(paths,sep=2):
    athdfs = []
    paths = [p for p in glob.glob(paths)]
    for i, path in enumerate(paths):
        athdfs.append(athena_read.athdf(path))
        if i % sep == 0:
            print(f'{i}:Loading {path}')
    print(f'Total of {len(athdfs)} files loaded')
    return athdfs

# Parallel code for computing the S3 longitudinal structure
# Adapted from Natascha's code


def distance(x,y,boxl):
    #calculate the distance form point x=(x1,x2,x3) to point y=(y1,y2,y3)
    #boxl = 1.0
    sep = np.abs(x - y)
    dist = np.sqrt(np.sum((np.where(sep>boxl/2.,np.abs(sep-boxl),sep))**2,axis=-1))
    return dist

def S3longitudinal_Sampling_ByCoord(agrs):
    data, r, sample_coords = agrs['data'], agrs['r'], agrs['sample_coords']
    z,y,x=np.meshgrid(data['x3v'],data['x2v'],data['x1v'],indexing='ij')
    pos3D=np.stack((x,y,z),axis=-1)
    dx=data['x3f'][1]-data['x3f'][0]
    
    #build initial distance mask
    xc = pos3D[0,0,0]
    dist = distance(pos3D,xc,1.0)
    cond = ((dist>r-dx*0.6) & (dist<r+dx*0.6))
    where = np.where(cond)
    
    #print(pos3D.shape)
    
    s3sum = 0.0
    nitems = 0
    n3,n2,n1=data["RootGridSize"]
    
    assert all([coord < n1*n2*n3 for coord in sample_coords]), "The sampling coordinates should not exceed the number of grids!"
    
    for sample_coord in sample_coords:
        k,j,i = sample_coord % n3, (sample_coord//n3)%n2, ((sample_coord//n3)//n2)%n1
        
        xc = pos3D[k,j,i]
        vc=np.array([data['vel1'][k,j,i],data['vel2'][k,j,i],data['vel3'][k,j,i]])

        #roll mask to new position
        mask = ((where[0]+k)%n3,(where[1]+j)%n2 ,(where[2]+i)%n1)

        vcond = np.stack((data['vel1'][mask],data['vel2'][mask],data['vel3'][mask]),axis=-1)
        poscond = pos3D[mask]


        rel = poscond - xc
        lens = np.sqrt(np.sum(rel**2,axis=-1))
        erel = rel/lens[:,None]

        s3sum += np.sum(np.power(np.sum((vcond-vc)*erel,axis=-1),3.0))
        nitems +=poscond.shape[0]
        #print(nitems,"\n")
    
    return s3sum/float(nitems)

def make_chunks(arr, nchunks):
    assert nchunks <= len(arr), "The number of chunks cannot exceed the size of array"
    
    # Calculate the chunk size to split
    res = len(arr) % nchunks
    chunk_size = len(arr)//nchunks
    
    chunks = []
    start, end = 0,0
    for i in range(nchunks):
        window_size = chunk_size+1 if i < res else chunk_size
        
        end = start + window_size
        
        if i == nchunks -1:
            end = len(arr)
            
        chunks.append(arr[start:end])
        
        start += window_size
            
    return chunks


def S3longitudinal_Sampling_Multiproc(data,r,nsamples,nproc):
    n3,n2,n1=data["RootGridSize"]
    
    nchunks = nsamples//nproc
    sample_coords = np.random.choice(n1*n2*n3,nsamples)
    sample_chunks = make_chunks(sample_coords,nproc) # Break the coordinates into chunks 
    chunks_size = [len(chunk) for chunk in sample_chunks]

    map_agrs = [{'r':r,'data':data,'sample_coords':sample_chunk} for sample_chunk in sample_chunks]
    
    #print([len(map_agr['sample_coords']) for map_agr in map_agrs])
    #print(sum([len(map_agr['sample_coords']) for map_agr in map_agrs]))

    with multip.Pool(nproc) as pool:
        s3_stats = pool.map(S3longitudinal_Sampling_ByCoord,map_agrs)

    del map_agrs # Clean things up

    return sum([s3_stat*csize for s3_stat,csize in zip(s3_stats,chunks_size)])/nsamples

def write_line(filename, value):
    with open(filename,'a') as f:
        f.write(value+'\n')
def clear_file(filename):
    with open(filename,"w") as f:
        f.write("")

if __name__ == "__main__":
    main()