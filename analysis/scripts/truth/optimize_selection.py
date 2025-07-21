import scipy.optimize as opt
import numpy as np
from astropy.io.misc.hdf5 import read_table_hdf5
import optimizecuts as oc
from importlib import reload
reload(oc)

crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)

hsc_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/hsc_wide_mags_failure_flags.hdf5'
hsc_cat = read_table_hdf5(input=hsc_file_path)

# loop through different weights and print out the corresponding parameter outputs for color shifts / mins of 1.1 < z < 1.6 elgs 
weight = np.linspace(0.0003, 0.0012, 4)

for i, w in enumerate(weight):
    rishift_guess = 0 
    iyshift_guess = 0.06
    izmin_guess = 0.36
    gfiblim_guess = 24.25
    zrange = (1.1, 1.6)
    guess = np.array([rishift_guess, iyshift_guess, izmin_guess, gfiblim_guess])
    args = (combined_cat, hsc_cat, zrange, w)
    
    result = opt.minimize(oc.opt_wrapper_truth, guess, args=args, method='COBYLA', options={ 'disp': True})
    density = oc.get_surf_density_truth(hsc_cat, rishift=result.x[0] , iyshift=result.x[1], izmin=result.x[2], gfiblim=result.x[3])
    zsuccess, rangesuccess = oc.get_success_rate_truth(combined_cat, zrange=zrange, rishift =result.x[0] , iyshift=result.x[1], izmin=result.x[2], gfiblim=result.x[3])
    density_yield = density*rangesuccess
    print(f'weight {w} with rishift = {result.x[0]:0.05}, iyshift = {result.x[1]:0.05}, izmin = {result.x[2]:0.05} and gfiblim = {result.x[3]:0.05}, z success and z range success are {zsuccess:0.05} and {rangesuccess:0.05} with target density and surface density yield of {density:0.05} and {density_yield:0.05}')