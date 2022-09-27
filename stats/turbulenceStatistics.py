import numpy as np
from scipy import stats

""" Calculate the energy spectrum as a function of radius

Parameters
----------
data : dict
       Athena dictionary of primitive values
dust : bool, optional
       If the energy spectrum of the dust (True) or gas (False) component should be calculated. Default: False.
       
Returns
-------
bins  : ndarray
        1D array of frequency bins for which the spectrum was calculated
datab : ndarray
        1D array containing the energy spectrum
"""

def energySpectrum(data,dust=False):

    n1,n2,n3 =data['RootGridSize']
    x3 = data['x3v']
    d3 = data['x3f'][1]-data['x3f'][0]
    f3 = 1./d3
    
    if(dust):
        v1 = data['vp1']
        v2 = data['vp2']
        v3 = data['vp3']
        rho= data['rhop']
    else:
        v1 = data['vel1']
        v2 = data['vel2']
        v3 = data['vel3']
        rho= data['rho']

    fftv1 =np.fft.fftn(v1)/(n1*n2*n3)
    fftv2 =np.fft.fftn(v2)/(n1*n2*n3)
    fftv3 =np.fft.fftn(v3)/(n1*n2*n3)

    k3 = np.fft.fftfreq(n3, d3)
    #k3 = k3[:k3.size//2]
    k = np.sqrt(k3[:,None,None]*k3[:,None,None]+k3[None,:,None]*k3[None,:,None]+k3[None,None,:]*k3[None,None,:])
    #k.shape
    Ek = np.real(fftv1*fftv1.conjugate()+fftv2*fftv2.conjugate()+fftv3*fftv3.conjugate())
    
    kmax=np.ceil(np.amax(k))
    bins_edge = np.arange(1.0,kmax,1)-0.5
    bins = np.arange(1.0,kmax-1,1)

    espec = np.stack((k.flatten(),Ek.flatten()),axis=-1)
    #print data.shape
    datab,__,__ = stats.binned_statistic(espec[:,0],espec[:,1],bins=bins_edge,statistic='sum')
    
    return bins,datab/2.0
    
"""Calculates the vorticity

Calculates the components of the 3 dim. vorticity vector assuming periodic boundaries in all dimensions

Parameters
----------
data : dict
       Athena dictionary of primitive values

Returns
-------
vox, voy, voz : ndarray
                3 dim. arrays containing the x, y ,z component of vorticity, respecitively

"""    
    
def vorticity(data):
    
    #get box size
    L1 = data['RootGridX1'][2]
    L2 = data['RootGridX2'][2]
    L3 = data['RootGridX3'][2]
    
    #get cell center positions and add periodic boundary values
    x = np.append(np.insert(data['x1v'],0,data['x1v'][-1]-L1),data['x1v'][0]+L1)
    y = np.append(np.insert(data['x2v'],0,data['x2v'][-1]-L2),data['x2v'][0]+L2)
    z = np.append(np.insert(data['x3v'],0,data['x3v'][-1]-L3),data['x3v'][0]+L3)
    
    
    #get velocities and assign boundary values (periodic for now)
    n3,n2,n1=data["RootGridSize"]
    
    vx =np.empty((n1+2,n2+2,n3+2))
    vy =np.empty((n1+2,n2+2,n3+2))
    vz =np.empty((n1+2,n2+2,n3+2))
    
    vx[1:-1,1:-1,1:-1] = np.swapaxes(data['vel1'],0,2)
    vy[1:-1,1:-1,1:-1] = np.swapaxes(data['vel2'],0,2)
    vz[1:-1,1:-1,1:-1] = np.swapaxes(data['vel3'],0,2)
    
    vx[0,:,:]  = vx[-2,:,:]
    vx[-1,:,:] = vx[1,:,:]
    vy[:,0,:]  = vy[:,-2,:]
    vy[:,-1,:] = vy[:,1,:]
    vz[:,:,0]  = vz[:,:,-2]
    vz[:,:,-1] = vz[:,:,1]
    
    #initialize vorticity arrays
    
    vox = np.zeros(tuple(data["RootGridSize"]))
    voy = np.zeros(tuple(data["RootGridSize"]))
    voz = np.zeros(tuple(data["RootGridSize"]))
    
    #calculate vorticity using second order central derivatives inside domain
    vox = (vz[1:-1,2:,1:-1]-vz[1:-1,:-2,1:-1])/((y[2:]-y[:-2])[None,:,None]) \
        - (vy[1:-1,1:-1,2:]-vy[1:-1,1:-1,:-2])/((z[2:]-z[:-2])[None,None,:]) #dy vz -dz vy
    
    voy = (vx[1:-1,1:-1,2:]-vx[1:-1,1:-1,:-2])/((z[2:]-z[:-2])[None,None,:]) \
        - (vz[2:,1:-1,1:-1]-vz[:-2,1:-1,1:-1])/((x[2:]-x[:-2])[:,None,None]) #dz vx -dx vz
    
    voz = (vy[2:,1:-1,1:-1]-vy[:-2,1:-1,1:-1])/((x[2:]-x[:-2])[:,None,None]) \
        - (vx[1:-1,2:,1:-1]-vx[1:-1,:-2,1:-1])/((y[2:]-y[:-2])[None,:,None]) #dx vy -dy vx
    
    return vox,voy,voz
    
    
"""Calculate Kolmogorov time microscale
   
   Calculate time scale corresponding to the Kolmogorove microscale using the full Strain-Rate tensor.

Parameters
----------
data : dict
       Athena dictionary of primitive values

Returns
-------
tscale : float
         Time mircoscale


"""
def tMicroscale(data):
    
    #get box size
    L1 = data['RootGridX1'][2]
    L2 = data['RootGridX2'][2]
    L3 = data['RootGridX3'][2]
    
    #get cell center positions and add periodic boundary values
    x = np.append(np.insert(data['x1v'],0,data['x1v'][-1]-L1),data['x1v'][0]+L1)
    y = np.append(np.insert(data['x2v'],0,data['x2v'][-1]-L2),data['x2v'][0]+L2)
    z = np.append(np.insert(data['x3v'],0,data['x3v'][-1]-L3),data['x3v'][0]+L3)
    
    
    #get velocities and assign boundary values (periodic for now)
    n3,n2,n1=data["RootGridSize"]
    
    vx =np.empty((n1+2,n2+2,n3+2))
    vy =np.empty((n1+2,n2+2,n3+2))
    vz =np.empty((n1+2,n2+2,n3+2))
    
    vx[1:-1,1:-1,1:-1] = np.swapaxes(data['vel1'],0,2)
    vy[1:-1,1:-1,1:-1] = np.swapaxes(data['vel2'],0,2)
    vz[1:-1,1:-1,1:-1] = np.swapaxes(data['vel3'],0,2)
    
    vx[0,:,:]  = vx[-2,:,:]
    vx[-1,:,:] = vx[1,:,:]
    vy[:,0,:]  = vy[:,-2,:]
    vy[:,-1,:] = vy[:,1,:]
    vz[:,:,0]  = vz[:,:,-2]
    vz[:,:,-1] = vz[:,:,1]
    
    #initialize Strain rate tensor array (only 6 components as tensor is symmetric)
    
    E = np.zeros((6,n1,n2,n3))
    
    #calculate vorticity using second order central derivatives
    
    E[0,:,:,:] = (vx[2:,1:-1,1:-1]-vx[:-2,1:-1,1:-1])/((x[2:]-x[:-2])[None,None,:]) #E11 = dx vx
    
    E[1,:,:,:] = (vy[1:-1,2:,1:-1]-vy[1:-1,:-2,1:-1])/((y[2:]-y[:-2])[:,None,None]) #E22 = dy vy
    
    E[2,:,:,:] = (vz[1:-1,1:-1,2:]-vz[1:-1,1:-1,:-2])/((z[2:]-z[:-2])[None,:,None]) #E33 = dz vz
    
    E[3,:,:,:] = 0.5*((vy[2:,1:-1,1:-1]-vy[:-2,1:-1,1:-1])/((x[2:]-x[:-2])[:,None,None])
                          + (vx[1:-1,2:,1:-1]-vx[1:-1,:-2,1:-1])/((y[2:]-y[:-2])[None,:,None])) #E12=E21= dx vy +dy vx
    
    E[4,:,:,:] = 0.5*((vx[1:-1,1:-1,2:]-vx[1:-1,1:-1,:-2])/((z[2:]-z[:-2])[None,None,:])
                          + (vz[2:,1:-1,1:-1]-vz[:-2,1:-1,1:-1])/((x[2:]-x[:-2])[:,None,None])) #E13=E31= dz vx +dx vz
    
    E[5,:,:,:] = 0.5*((vz[1:-1,2:,1:-1]-vz[1:-1,:-2,1:-1])/((y[2:]-y[:-2])[None,:,None])
                          + (vy[1:-1,1:-1,2:]-vy[1:-1,1:-1,:-2])/((z[2:]-z[:-2])[None,None,:])) #E23=E32= dy vz +dz vy
    
    
    tinv = np.sqrt(2.0*np.mean(E[0]*E[0]+E[1]*E[1]+E[2]*E[2]+2*E[3]*E[3]+2*E[4]*E[4]+2*E[5]*E[5]))
    
    return 1.0/tinv
    
 
""" Periodic Euclidean distance metric

Calculates the euclidean distance between point x and point y assuming a periodic box with length boxl in all directions

Parameters
----------
x,y  : ndarray
       Points of which to calculate the mutual distance, trailing axes dimension has to match
boxl : float
       Length of the periodic box

Returns
-------
dist : float
       distance between points
 
"""
def distance(x,y,boxl):
    #calculate the distance form point x=(x1,x2,x3) to point y=(y1,y2,y3)
    #boxl = 1.0
    sep = np.abs(x - y)
    dist = np.sqrt(np.sum((np.where(sep>boxl/2.,np.abs(sep-boxl),sep))**2,axis=-1))
    return dist
    

 
"""Calculate the 3rd order longitudinal structure function for specific distance

Parameters
----------
data : dict
       Athena dictionary of primitive values

r    : float
       correlation distance 
       
Returns
-------
s3long : float

""" 
def S3longitudinal(data,r):
    z,y,x=np.meshgrid(data['x3v'],data['x2v'],data['x1v'],indexing='ij')
    pos3D=np.stack((x,y,z),axis=-1)
    dx=data['x3f'][1]-data['x3f'][0]
    
    #build initial distance mask
    xc = pos3D[0,0,0]
    dist = distance(pos3D,xc,1.0)
    cond = ((dist>r-dx*0.6) & (dist<r+dx*0.6))
    where = np.where(cond)
    
    s3sum = 0.0
    nitems = 0
    n3,n2,n1=data["RootGridSize"]
    for k in range(0,n3):
        for j in range(0,n2):
            for i in range(0,n1):
                #
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