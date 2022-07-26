import glob
import numpy as np
from scipy import stats
import argparse 
import matplotlib.pyplot as plt
from datetime import datetime
import json 

def main():
    args = get_args()
    rho_paths = sorted(list(glob.glob(args.rhop)))

    rdf_out = []
    for path in rho_paths:
        rhop = np.load(path).squeeze()
        dcf = dcf_fft(rhop)

        bin_centers, bin_edges = make_geom_bins(1,250,35)
        dcf_radial = radial_binning(dcf,bin_edges)

        bin_centers *= args.boxl[0]/rhop.shape[0] # Rescale the length
        bin_edges   *= args.boxl[0]/rhop.shape[0]

        out = {'rho-path':path,'binEdges':bin_edges.tolist(),'binCenters':bin_centers.tolist(),'DCF':dcf_radial.tolist(),'RDF':(dcf_radial+1).tolist()}
        
        rdf_out.append(out)

    now = datetime.now()
    timestamp = int(now.timestamp())
    
    with open(f'{args.outname}-{timestamp}.json','w') as result:
        json.dump(rdf_out,result)



def make_geom_bins(begin,end,num=50):
    bin_edges = np.geomspace(begin,end,num=num+1)
    bin_centers = np.sqrt(bin_edges[:-1]*bin_edges[1:])
    return bin_centers, bin_edges

def dcf_fft(n):
    N1,N2,N3 = n.shape
    n_bar = np.mean(n)
    n_tdm1 = n/n_bar -1 # n tilde - 1
    fft_n_tdm1 = np.fft.fftn(n_tdm1)
    #print(np.conj(fft_n_tdm1),fft_n_tdm1)
    dcf = np.fft.ifftn(np.conj(fft_n_tdm1)*fft_n_tdm1)
    return np.real(dcf)/(N1*N2*N3)

def R2_like(field):
    axes = [np.fft.fftfreq(dim,d=1/dim) for dim in field.shape]
    if len(axes) == 3:
        axes=axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:]
    else:
        axes=axes[0][:,None],axes[1][None,:]
    return np.sum(np.array([ax**2 for ax in axes],dtype=object)) # Create radial axes


def radial_binning(field,bins): # Give radial average of field. Assume input field unshifted
    R = np.sqrt(R2_like(field))
    
    radial_bin, _, _ = stats.binned_statistic(R.flatten(),field.flatten(),statistic='mean',bins=bins)
    return radial_bin


def get_args():
    parser = argparse.ArgumentParser(description='Radial distribution function (RDF) using FFT')
    parser.add_argument('--rhop',type=str,required=True,help='Glob to rhop (npy)')
    parser.add_argument('--boxl',type=list,help='Box size in simulation unit',default=[1.0,1.0,1.0])
    parser.add_argument('--outname',type=str,help='Output file name. Timestamp always added',default='RDF-FFT-GRID')

    return parser.parse_args()

if __name__ == "__main__":
    main()
