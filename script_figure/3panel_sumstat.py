import colorcet as cc
import astropy
import scipy.optimize as opt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.table import Table 
from astropy import units as u 
from astropy import constants as const
from astropy import table
from astropy.table import join, vstack, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.cm import get_cmap

#A second function might take an hsc_cat as input (as well as shifts and magnitude limits) and output surface densities
def surf_density(cat_full, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the surface density'''
    #define the color cuts
    rishift_min = np.minimum(0,rishift)
    cuts_iyri = np.logical_and((cat_full['r_mag'] - cat_full['i_mag'] < cat_full['i_mag'] - cat_full['y_mag'] - 0.19 + rishift_min),
                           (cat_full['i_mag'] - cat_full['y_mag'] > 0.35 + iyshift)) 
    highzcut = (cat_full['i_mag'] - cat_full['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)
    
    #define the magnitude cuts
    gfib_mask = cat_full['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(cat_full['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final = np.logical_and(colorcuts, gbandcut)
    cutcat = cat_full[cuts_final]
    #calculate the surface density
    area = 16
    surf_density = len(cutcat)/area
    return surf_density



def success_rate(catalog, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the success rate and redshift range success rate'''
    #define the color cuts
    cuts_iyri = np.logical_and((catalog['r_mag'] - catalog['i_mag'] < catalog['i_mag'] - catalog['y_mag'] - 0.19 + rishift),
                           (catalog['i_mag'] - catalog['y_mag'] > 0.35 + iyshift)) 
    highzcut = (catalog['i_mag'] - catalog['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)
    #exposure cuts
    exposure = catalog['TSNR2_LRG']*12.15
    tmask = exposure < 1400
    #define the magnitude cuts
    gfib_mask = catalog['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(catalog['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final_200 = np.logical_and(colorcuts, gbandcut)
    cuts_final = np.logical_and(cuts_final_200, tmask)
    cutcat_200 = catalog[cuts_final_200]
    cutcat = catalog[cuts_final]
    #calculate the success rate
    o2_snr = cutcat['OII_FLUX']*np.sqrt(cutcat['OII_FLUX_IVAR'])   
    chi2 = cutcat['DELTACHI2']
    o2_snr_200 = catalog['OII_FLUX'][cuts_final_200]*np.sqrt(catalog['OII_FLUX_IVAR'][cuts_final_200])   
    chi2_200 = catalog['DELTACHI2'][cuts_final_200]
    #masking specz catalog
    snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
    snr_mask = np.logical_or(snr_mask, chi2 > 25)
    snr_mask_200 = o2_snr_200 > 10**(0.9 - 0.2*np.log10(chi2_200))
    snr_mask_200 = np.logical_or(snr_mask_200, chi2_200 > 25)
    #calculate the fraction of objects with good redshifts with the 200 < t < 1400 sample 
    success_rate_200_1400 = np.sum(snr_mask)/np.sum(cuts_final)
    #calculate the redshift range success rate with 200 < t sample
    redshift_range_200 = np.sum(np.logical_and(snr_mask_200, np.logical_and(cutcat_200['Z'] > 1.05, cutcat_200['Z'] < 1.65)))/np.sum(cuts_final_200)
    #Calculate the net redshift yield
    redshift_range = redshift_range_200*success_rate_200_1400
    return success_rate_200_1400, redshift_range

#load in catalogs 
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
dir_out = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
hsc_cat = table.Table.read(dir_in+'HSC.fits',format='fits',hdu=1)

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
mask &= (~hsc_cat["i_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["i_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["g_cmodel_flux"]) & (hsc_cat["g_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["g_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["g_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["r_cmodel_flux"]) & (hsc_cat["r_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["r_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["r_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["y_cmodel_flux"]) & (hsc_cat["y_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["y_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["y_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["z_cmodel_flux"]) & (hsc_cat["z_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["z_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["z_sdsscentroid_flag"])
hsc_cat = hsc_cat[mask]


#load in specz cataloges 
tert = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/all_elgs.fits',format='fits',hdu=1)

#cleaning specz catalog
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0 
exposure = tert['TSNR2_LRG']*12.15
tmask = exposure > 200
ysh = tert['YSH'] == True
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, ysh))
'''START TO COMPARE TO FINAL CLEAN SELECTION, USE ELGS'''
elgs = tert[t_mask]
print(len(tert))
print('the number of elgs is', len(elgs))

#merge both hsc_cat and elgs catalogs, this combined catalog will be used to tweak the cuts and check our redshift distribution
ra_hsc = hsc_cat['ra']
dec_hsc = hsc_cat['dec']
ra_elg = elgs['TARGET_RA']
dec_elg = elgs['TARGET_DEC']
hsc_coord = SkyCoord(ra_hsc*u.degree, dec_hsc*u.degree)
elg_coord = SkyCoord(ra_elg*u.degree, dec_elg*u.degree)
idx_h, d2d_h, d3d_h = elg_coord.match_to_catalog_sky(hsc_coord)
dmask = d2d_h.arcsec < 0.000001
combined_cat = hstack([elgs, hsc_cat[idx_h]])[dmask]

#Masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat['OII_FLUX']*np.sqrt(combined_cat['OII_FLUX_IVAR'])   
chi2_comb = combined_cat['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail =~ snr_mask_comb


#Colors
ri = combined_cat['r_mag'][snr_mask_comb] -  combined_cat['i_mag'][snr_mask_comb]
iz = combined_cat['i_mag'][snr_mask_comb] -  combined_cat['z_mag'][snr_mask_comb]
iy = combined_cat['i_mag'][snr_mask_comb] -  combined_cat['y_mag'][snr_mask_comb]
ri_fail= combined_cat['r_mag'][snr_mask_comb_fail] -  combined_cat['i_mag'][snr_mask_comb_fail]
iz_fail = combined_cat['i_mag'][snr_mask_comb_fail] -  combined_cat['z_mag'][snr_mask_comb_fail]
iy_fail = combined_cat['i_mag'][snr_mask_comb_fail] -  combined_cat['y_mag'][snr_mask_comb_fail]
gband = combined_cat['g_mag'][snr_mask_comb]
gfiber = combined_cat['g_fiber_mag'][snr_mask_comb]
rband = combined_cat['r_mag'][snr_mask_comb]
rfiber = combined_cat['r_fiber_mag'][snr_mask_comb]

#Specz combined catalog
specz = combined_cat['Z'][snr_mask_comb]

#no cut version
ri_full = combined_cat['r_mag'] -  combined_cat['i_mag']
iz_full = combined_cat['i_mag'] -  combined_cat['z_mag']
iy_full = combined_cat['i_mag'] -  combined_cat['y_mag']
gband_full = combined_cat['g_mag']
gfiber_full = combined_cat['g_fiber_mag']
rband_full = combined_cat['r_mag']
rfiber_full = combined_cat['r_fiber_mag']

#Specz combined catalog
specz_full = combined_cat['Z']
print(len(combined_cat))

#define the arrays to fill with our results
nmag = 60
maglim= np.linspace(23,24.6,nmag)
zsuccessf = np.zeros(nmag)
rangesuccessf = np.zeros(nmag)
surface_densityf = np.zeros(nmag)
surf_density_actf = np.zeros(nmag)

#1.05 < z <  1.65 rioffset, iyoffset, izmin, and gfiblim values
rishift = 0 
iyshift = 0.06636818841724967
izmin = 0.36442580263398095
gfiblim = 24.263228615646897


for i, mlim in enumerate(maglim):
      ratesf = success_rate(combined_cat, rishift=rishift, iyshift=iyshift, izmin=izmin, gfiblim=mlim)
      surff = surf_density(hsc_cat, rishift=rishift, iyshift=iyshift, izmin=izmin, gfiblim=mlim)
      zsuccessf[i] = ratesf[0]
      rangesuccessf[i] = ratesf[1]
      surface_densityf[i] = surff
      surf_density_actf[i] = surff*rangesuccessf[i]

      #Here, we are going to plot a panel with the target density, net surface density yield, and redshift range success rate 
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(29, 10), tight_layout=True)
ax0.plot(maglim, surface_densityf, color = 'black')
ax0.set_xlabel('g Fiber magnitude limit', fontsize = 25)
ax0.set_ylabel(r'Target density ($\rho_\mathrm{target}$)', fontsize = 25)
ax0.plot(24.21, 1930, marker = '*', color = 'red', markersize = 19, label = 'DESI ELG LOP sample')
ax0.axvline(x=gfiblim,ls= '--',color='purple', label = '$ 1.05 < z < 1.65$ sample mag cutoff')
ax0.xaxis.set_tick_params(labelsize = 18)
ax0.yaxis.set_tick_params(labelsize = 18)
ax0.legend(loc = 'upper left', fontsize = 22)
# '$z_{\mathrm{spec}}$'
#& $f_\mathrm{reliable}$ & $f_\mathrm{range yield}$ & $\rho_\mathrm{target}$ & $\rho_\mathrm{yield}$

ax1.plot(maglim, surf_density_actf, color = 'blue')
ax1.set_xlabel('g Fiber magnitude limit', fontsize = 25)
ax1.set_ylabel(r'Net surface density yield ($\rho_\mathrm{yield}$)', fontsize = 25)
ax1.xaxis.set_tick_params(labelsize = 18)
ax1.yaxis.set_tick_params(labelsize = 18)
ax1.axvline(x=gfiblim,ls= '--',color='purple')
ax1.plot(24.21, 428, marker = '*', color = 'red', markersize = 19)


ax2.plot(maglim, rangesuccessf, color = 'orange')
ax2.set_xlabel('g Fiber magnitude limit', fontsize = 25)
ax2.set_ylabel(r'Redshift range success rate ($f_\mathrm{range yield}$)', fontsize = 25)
ax2.set_ylim(0,1)
ax2.xaxis.set_tick_params(labelsize = 18)
ax2.yaxis.set_tick_params(labelsize = 18)
ax2.axvline(x=gfiblim,ls= '--',color='purple')
ax2.plot(24.21, 0.32, marker = '*', color = 'red', markersize = 19)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/3panel_sumstat.png')


