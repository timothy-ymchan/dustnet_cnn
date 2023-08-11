import numpy as np
import time
import multiprocessing as multip
from multiprocessing.shared_memory import SharedMemory
from scipy import stats
import glob
import athena_read
import h5py
import matplotlib.pyplot as plt
import argparse

def main(args):

    """normal = read_athdf('../athenaMLdata/data/*103*')
    rs = np.geomspace(1/128,1,10)
    nproc=5
    for i,r in enumerate(rs):
        print('Job ',i+1)
        print('Number of processors: ',nproc)
        st = time.time()
        s3_val = S3longitudinal_Sampling_Multiproc(data=normal[0],r=r,nsamples=1000000,nproc=nproc,memshare=memshare) # 1000000 samples run
        write_line('./s3long-out/s3-normal-sampling-1000000-50-pts.txt',f"{r},{s3_val}") 
        et = time.time()
        print(f'Processing time: {et-st} sec')
        if i % 5 == 0:
            print('Iteration: ',i+1,'r=',r)"""
    # Print configurations
    print('Input file: ',args.in_path)
    print('Output prefix: ',args.out_prefix)
    print('Memory sharing: ',args.memshare)
    print('r values: ',args.r_bins)
    in_file = read_athdf(args.in_path)
    r_bins = [eval(r) for r in args.r_bins.split(',')]
    #rint(r_bins)
    rs = np.geomspace(*r_bins)
    filename = f'{args.out_prefix}-sampling-{args.nsamples}-{len(rs)}-pts.txt'

    write_line(filename,f'# {args.in_path}\tsampling-{args.nsamples}\t{len(rs)} points')
    for i, r in enumerate(rs):
        print('Job ',i+1)
        st = time.time()
        s3_val = S3longitudinal_Sampling_Multiproc(data=in_file[0],r=r,nsamples=args.nsamples,nproc=args.nproc,memshare=args.memshare) # 1000000 samples run
        write_line(filename,f"{r},{s3_val}") 
        print('The estimated statistics at r=',r,' is S3=',s3_val)
        et = time.time()
        print(f'Processing time: {et-st} sec')



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

def S3longitudinal_Sampling_ByCoord_SharedMemory(args):
    sample_coords = args['sample_coords']
    n3,n2,n1 = args['RootGridSize']

    # Load all the shared memory
    spos3D = SharedMemory(name=args['pos3D']['name'])
    pos3D = np.ndarray(shape=args['pos3D']['shape'],dtype=args['pos3D']['dtype'],buffer=spos3D.buf)
    
    svel1 = SharedMemory(name=args['vel1']['name'])
    vel1 = np.ndarray(shape=args['vel1']['shape'],dtype=args['vel1']['dtype'],buffer=svel1.buf)

    svel2 = SharedMemory(name=args['vel2']['name'])
    vel2 = np.ndarray(shape=args['vel2']['shape'],dtype=args['vel2']['dtype'],buffer=svel2.buf)

    svel3 = SharedMemory(name=args['vel3']['name'])
    vel3 = np.ndarray(shape=args['vel3']['shape'],dtype=args['vel3']['dtype'],buffer=svel3.buf)

    swhere = SharedMemory(name=args['where']['name'])
    where = np.ndarray(shape=args['where']['shape'],dtype=args['where']['dtype'],buffer=swhere.buf)
    where = where.tolist()

    s3sum = 0.0
    nitems = 0
    
    assert all([coord < n1*n2*n3 for coord in sample_coords]), "The sampling coordinates should not exceed the number of grids!"
    
    for sample_coord in sample_coords:
        k,j,i = sample_coord % n3, (sample_coord//n3)%n2, ((sample_coord//n3)//n2)%n1
        
        xc = pos3D[k,j,i]
        vc=np.array([vel1[k,j,i],vel2[k,j,i],vel3[k,j,i]])

        #roll mask to new position
        mask = ((where[0]+k)%n3,(where[1]+j)%n2 ,(where[2]+i)%n1)

        vcond = np.stack((vel1[mask],vel2[mask],vel3[mask]),axis=-1)
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


def S3longitudinal_Sampling_Multiproc(data,r,nsamples,nproc,memshare=False):
    n3,n2,n1=data["RootGridSize"]
    
    nchunks = nsamples//nproc
    sample_coords = np.random.choice(n1*n2*n3,nsamples)
    sample_chunks = make_chunks(sample_coords,nproc) # Break the coordinates into chunks 
    chunks_size = [len(chunk) for chunk in sample_chunks]
    
    if not memshare:
        map_args = [{'r':r,'data':data,'sample_coords':sample_chunk} for sample_chunk in sample_chunks]
    
        #print([len(map_agr['sample_coords']) for map_agr in map_agrs])
        #print(sum([len(map_agr['sample_coords']) for map_agr in map_agrs]))

        with multip.Pool(nproc) as pool:
            s3_stats = pool.map(S3longitudinal_Sampling_ByCoord,map_args)
        
    else:
        # Create shared memory for vel1, vel2, vel3, pos3D, where
        z,y,x=np.meshgrid(data['x3v'],data['x2v'],data['x1v'],indexing='ij')
        pos3D=np.stack((x,y,z),axis=-1)
        dx=data['x3f'][1]-data['x3f'][0]

        #Precompute the distance mask
        xc = pos3D[0,0,0]
        dist = distance(pos3D,xc,1.0)
        cond = ((dist>r-dx*0.6) & (dist<r+dx*0.6))
        where = np.array(np.where(cond))
        
        spos3D = Make_SharedMemory(pos3D,name='pos3D')
        svel1 = Make_SharedMemory(data['vel1'],name='vel1')
        svel2 = Make_SharedMemory(data['vel2'],name='vel2')
        svel3 = Make_SharedMemory(data['vel3'],name='vel3')
        swhere = Make_SharedMemory(where,name='where')

        map_args = [{'r':r,'RootGridSize':(n3,n2,n1),'pos3D':spos3D,'vel1':svel1,'vel2':svel2,'vel3':svel3,'where':swhere,'sample_coords':sample_chunk} for sample_chunk in sample_chunks]
        
    
        with multip.Pool(nproc) as pool:
            s3_stats = pool.map(S3longitudinal_Sampling_ByCoord_SharedMemory,map_args)
        
        # Free the shared memory
        Close_SharedMemory('pos3D')
        Close_SharedMemory('vel1')
        Close_SharedMemory('vel2')
        Close_SharedMemory('vel3')
        Close_SharedMemory('where')

    del map_args

    return sum([s3_stat*csize for s3_stat,csize in zip(s3_stats,chunks_size)])/nsamples


#def sizeof(np_arr):
#    # Return size of numpy array in bytes
#    return np_arr.size * np_arr.itemsize

def Make_SharedMemory(numpy_arr,name):
    sm = SharedMemory(name=name,create=True,size=numpy_arr.nbytes)
    arr = np.ndarray(shape=numpy_arr.shape,dtype=numpy_arr.dtype,buffer=sm.buf)
    np.copyto(arr,numpy_arr) # Copy array results to shared memory
    return {'name':name, 'shape':numpy_arr.shape, 'dtype':numpy_arr.dtype} # Return meta information

def Close_SharedMemory(name):
    sm = SharedMemory(name=name)
    sm.close()
    sm.unlink()
#def Call_SharedMemory(numpy_arr,name):


def write_line(filename, value):
    with open(filename,'a') as f:
        f.write(value+'\n')
def clear_file(filename):
    with open(filename,"w") as f:
        f.write("")

if __name__ == "__main__":
    # Handle all the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--memshare',help='Use shared memory',default=False,type=bool)
    parser.add_argument('--in-path',help='Input file glob',required=True,type=str)
    parser.add_argument('--out-prefix',help='Output prefix',required=True,type=str)
    parser.add_argument('--r-bins',help='Geometric bin arguments for r in format (r0,r1,N)',required=True,type=str)
    parser.add_argument('--nproc',help='Number of processors',required=False,default=5,type=int)
    parser.add_argument('--nsamples',help='Number of sample points',required=False,default=50000)

    args = parser.parse_args()
    
    main(args)