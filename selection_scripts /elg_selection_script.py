import os
import time
import numpy as np 
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import table
from astropy.table import join
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt 

#Load the catalog 
catversion = 'Farmer'  # this string can be either 'Classic' or 'Farmer'
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
dir_out = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
hsc_cat = table.Table.read(dir_in+'HSC.fits'.format(catversion),format='fits',hdu=1).to_pandas()

def flux_to_mag(flux):
    return -2.5*np.log10(flux*1e-9) + 8.90
# extinction corrected mags (extinction is negligible for XMM-LSS)
hsc_cat["i_mag"] = flux_to_mag(hsc_cat["i_cmodel_flux"])-hsc_cat["a_i"]
hsc_cat["r_mag"] = flux_to_mag(hsc_cat["r_cmodel_flux"])-hsc_cat["a_r"]
hsc_cat["z_mag"] = flux_to_mag(hsc_cat["z_cmodel_flux"])-hsc_cat["a_z"]
hsc_cat["g_mag"] = flux_to_mag(hsc_cat["g_cmodel_flux"])-hsc_cat["a_g"]
hsc_cat["y_mag"] = flux_to_mag(hsc_cat["y_cmodel_flux"])-hsc_cat["a_y"]


hsc_cat["i_fiber_mag"] = flux_to_mag(hsc_cat["i_fiber_flux"])-hsc_cat["a_i"]
hsc_cat["i_fiber_tot_mag"] = flux_to_mag(hsc_cat["i_fiber_tot_flux"])-hsc_cat["a_i"]
hsc_cat["g_fiber_mag"] = flux_to_mag(hsc_cat["g_fiber_flux"])-hsc_cat["a_g"]
hsc_cat["g_fiber_tot_mag"] = flux_to_mag(hsc_cat["g_fiber_tot_flux"])-hsc_cat["a_g"]
hsc_cat["r_fiber_mag"] = flux_to_mag(hsc_cat["r_fiber_flux"])-hsc_cat["a_r"]
hsc_cat["r_fiber_tot_mag"] = flux_to_mag(hsc_cat["r_fiber_tot_flux"])-hsc_cat["a_r"]

## Quality cuts
# valid I-band flux
mask = np.isfinite(hsc_cat["i_cmodel_flux"]) & (hsc_cat["i_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["i_cmodel_flag"].values)
#General Failure Flag
mask &= (~hsc_cat["i_sdsscentroid_flag"].values)
mask &= np.isfinite(hsc_cat["g_cmodel_flux"]) & (hsc_cat["g_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["g_cmodel_flag"].values)
#General Failure Flag
mask &= (~hsc_cat["g_sdsscentroid_flag"].values)
mask &= np.isfinite(hsc_cat["r_cmodel_flux"]) & (hsc_cat["r_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["r_cmodel_flag"].values)
#General Failure Flag
mask &= (~hsc_cat["r_sdsscentroid_flag"].values)
mask &= np.isfinite(hsc_cat["y_cmodel_flux"]) & (hsc_cat["y_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["y_cmodel_flag"].values)
#General Failure Flag
mask &= (~hsc_cat["y_sdsscentroid_flag"].values)
mask &= np.isfinite(hsc_cat["z_cmodel_flux"]) & (hsc_cat["z_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["z_cmodel_flag"].values)
#General Failure Flag
mask &= (~hsc_cat["z_sdsscentroid_flag"].values)
hsc_cat = hsc_cat[mask]

#Adding the adjusted HSC mags to the HSC catalog
hsc_cat["g_mag_cosmos"] = hsc_cat["g_mag"] + 0.10
hsc_cat["r_mag_cosmos"] = hsc_cat["r_mag"] + 0.11
hsc_cat["i_mag_cosmos"] = hsc_cat["i_mag"] + 0.11
hsc_cat["z_mag_cosmos"] = hsc_cat["z_mag"] + 0.09
hsc_cat["y_mag_cosmos"] = hsc_cat["y_mag"] + 0.11

#Colors
ri_adj = hsc_cat['r_mag_cosmos'] -  hsc_cat['i_mag_cosmos']
iz_adj = hsc_cat['i_mag_cosmos'] -  hsc_cat['z_mag_cosmos']
iy_adj = hsc_cat['i_mag_cosmos'] -  hsc_cat['y_mag_cosmos']

# Final target selection using the hsc_cat catalog with the HSC adjusted mags to be made into a script
gfib_mask = hsc_cat['g_fiber_mag'] < 24.3
gband_adj = np.logical_and(hsc_cat['g_mag_cosmos'] < 24, gfib_mask)
rfib_mask = hsc_cat['r_fiber_mag'] < 24.3
rband_adj = np.logical_and(hsc_cat['r_mag_cosmos'] < 24, rfib_mask)
inlimit_adj = np.logical_or(rband_adj, gband_adj)
cuts_iyri_adj = np.logical_and((iy_adj > 0.35),(ri_adj < iy_adj- 0.1)) 
cuts_final_adj = np.logical_and(cuts_iyri_adj, inlimit_adj)

i_d = hsc_cat['object_id'][cuts_final_adj]
ra = hsc_cat['ra'][cuts_final_adj]
dec = hsc_cat['dec'][cuts_final_adj]
gmag = hsc_cat["g_mag_cosmos"][cuts_final_adj]
rmag = hsc_cat["r_mag_cosmos"][cuts_final_adj]
imag = hsc_cat["i_mag_cosmos"][cuts_final_adj]
zmag = hsc_cat["z_mag_cosmos"][cuts_final_adj]
ymag = hsc_cat["y_mag_cosmos"][cuts_final_adj]

d = {'ID_HSC': i_d, 'RA_HSC': ra, 'DEC_HSC': dec, 'gmag_HSC_offset': gmag, 'rmag_HSC_offset':rmag, 'imag_HSC_offset':imag, 'zmag_HSC_offset':zmag, 'ymag_HSC_offset':ymag}
df = pd.DataFrame(data=d)
df
#Write an if statement to check if the file exists and if it does, append the new data to a new file
if os.path.isfile('ELG_DESI-2_target_list.csv'):
    df.to_csv('ELG_DESI-2_target_list.csv', mode='w', header= True, index = False)
# df.to_csv('ELG_DESI-2_target_list.csv', index = False)
print('The number of HSC ELG targets is', len(df))
