import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from astropy import table
from astropy.io.misc.hdf5 import read_table_hdf5


# the set of functions to use when optimizing using ELGs from Rongpu
def surf_density_rongpu(cat_full, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the surface density'''
    #define the color cuts
    cuts_iyri = np.logical_and((cat_full['r_mag'] - cat_full['i_mag'] < cat_full['i_mag'] - cat_full['y_mag'] - 0.19 + rishift),
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

# refactored the success_rate function
def success_rate_rongpu(catalog, zrange=(1.1, 1.6), ri_targcut=-0.19, iy_targcut= 0.35, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the success rate and redshift range success rate'''
    # define the color cuts
    cuts_iyri = np.logical_and(
        (catalog['r_mag'] - catalog['i_mag'] < catalog['i_mag'] - catalog['y_mag'] + ri_targcut + rishift),
        (catalog['i_mag'] - catalog['y_mag'] > iy_targcut + iyshift)) 
    highzcut = (catalog['i_mag'] - catalog['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)

    # define the magnitude cuts
    gfib_mask = catalog['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(catalog['g_mag'] < glim, gfib_mask)

    # combine the color and magnitude cuts
    # 1400 sec exposure cut
    exposure = catalog['EFFTIME']
    tmask = exposure < 1400
    cuts_final_700 = np.logical_and(colorcuts, gbandcut)
    

    # quality cuts for reliable speczs
    o2_snr = catalog['OII_FLUX']*np.sqrt(catalog['OII_FLUX_IVAR'])   
    chi2 = catalog['DELTACHI2']
    snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
    snr_o2_mask = np.logical_or(snr_mask, chi2 > 25) 


    # calculate the fraction of objects with good redshifts with the 700 < t < 1400 sample 
    success_700_1400 = np.sum(np.logical_and.reduce((cuts_final_700, tmask, snr_o2_mask)))/np.sum(np.logical_and(cuts_final_700, tmask))
    
    # calculate the redshift range success rate for t > 700 sample
    # this should be the ratio of the number of (objs at the correct redshift and with good speczs) to the (total number passing color/mag cuts and snr/o2 quality cut)
    
    zcut = np.logical_and(catalog['Z'] > zrange[0], catalog['Z'] < zrange[1])
    range_success_700 = np.sum(np.logical_and.reduce((cuts_final_700, snr_o2_mask, zcut)))/np.sum(np.logical_and(cuts_final_700, snr_o2_mask))
    
    # calculate the net redshift yield
    redshift_range_yield = range_success_700*success_700_1400

    denom = np.sum(np.logical_and(cuts_final_700, tmask))
    
    return success_700_1400, redshift_range_yield, denom 


# load hsc wide catalog
hsc_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/hsc_wide_mags_failure_flags.hdf5'
hsc_cat = read_table_hdf5(input=hsc_file_path)

# load in crossmatched cat of hsc wide and spec truth
crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)

# masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat['OII_FLUX']*np.sqrt(combined_cat['OII_FLUX_IVAR'])   
chi2_comb = combined_cat['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail = ~ snr_mask_comb


# colors
ri = combined_cat['r_mag'][snr_mask_comb] - combined_cat['i_mag'][snr_mask_comb]
iz = combined_cat['i_mag'][snr_mask_comb] - combined_cat['z_mag'][snr_mask_comb]
iy = combined_cat['i_mag'][snr_mask_comb] - combined_cat['y_mag'][snr_mask_comb]

ri_fail = combined_cat['r_mag'][snr_mask_comb_fail] - combined_cat['i_mag'][snr_mask_comb_fail]
iz_fail = combined_cat['i_mag'][snr_mask_comb_fail] - combined_cat['z_mag'][snr_mask_comb_fail]
iy_fail = combined_cat['i_mag'][snr_mask_comb_fail] - combined_cat['y_mag'][snr_mask_comb_fail]

gband = combined_cat['g_mag'][snr_mask_comb]
gfiber = combined_cat['g_fiber_mag'][snr_mask_comb]
rband = combined_cat['r_mag'][snr_mask_comb]
rfiber = combined_cat['r_fiber_mag'][snr_mask_comb]

# specz combined catalog
specz = combined_cat['Z'][snr_mask_comb]

# no cut version
ri_full = combined_cat['r_mag'] - combined_cat['i_mag']
iz_full = combined_cat['i_mag'] - combined_cat['z_mag']
iy_full = combined_cat['i_mag'] - combined_cat['y_mag']

gband_full = combined_cat['g_mag']
gfiber_full = combined_cat['g_fiber_mag']
rband_full = combined_cat['r_mag']
rfiber_full = combined_cat['r_fiber_mag']

# specz with no cuts
specz_full = combined_cat['Z']
print(f'The length of the combined catalog is: {len(combined_cat)}')

# define the arrays to fill with our results
nmag = 60
maglim = np.linspace(23, 24.6, nmag)
zsuccessf = np.zeros(nmag)
rangesuccessf = np.zeros(nmag)
surface_densityf = np.zeros(nmag)
surf_density_actf = np.zeros(nmag)

# best case for 1.1 < z <  1.6 for rioffset, iyoffset, izmin, and gfiblim values
optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path)
rishift, iyshift, izmin, gfiblim = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]



for i, mlim in enumerate(maglim):
    ratesf = success_rate_rongpu(combined_cat, rishift=rishift, iyshift=iyshift, izmin=izmin, gfiblim=mlim)
    surff = surf_density_rongpu(hsc_cat, rishift=rishift, iyshift=iyshift, izmin=izmin, gfiblim=mlim)
    zsuccessf[i] = ratesf[0]
    rangesuccessf[i] = ratesf[1]
    surface_densityf[i] = surff
    surf_density_actf[i] = surff * rangesuccessf[i]

# here, we are going to plot a panel with the target density, net surface density yield, and redshift range success rate


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(29, 10), tight_layout=True)
plt.subplots_adjust(wspace=0.82)

ax0.plot(maglim, rangesuccessf, color = 'black', lw=4, zorder=1)
ax0.set_xlabel(r'$g_\mathrm{fiber}$ mag limit', fontsize = 32)
ax0.set_ylabel(r'Redshift range success rate ($f_\mathrm{z\,= \, 1.1\, -\, 1.6}^{700 \,<\,t\, <\,1400}$)', fontsize = 31)
ax0.set_ylim(0,1)
ax0.xaxis.set_tick_params(labelsize = 29)
ax0.yaxis.set_tick_params(labelsize = 29)
ax0.axvline(x=gfiblim,ls= '--',color='purple', lw=4, zorder=2)
ax0.plot(24.21, 0.32, marker = '*', color = 'grey', markersize = 30, linestyle='None', zorder=3)
ax0.plot(gfiblim, 0.84, marker = '*', color = 'blue', markersize = 30, linestyle='None', zorder=4)

ax1.plot(maglim, surface_densityf, color = 'black', lw=4, zorder=1)
ax1.set_xlabel(r'$g_\mathrm{fiber}$ mag limit', fontsize = 32)
ax1.set_ylabel(r'Target density ($\Sigma_\mathrm{target}$)', fontsize = 31)
ax1.plot(24.21, 1930, marker = '*', color = 'grey', markersize = 30, label = 'DESI ELG LOP', linestyle='None')
ax1.axvline(x=gfiblim,ls= '--',color='purple', label = 'Our g-fiber limit', lw=4, zorder=2)
ax1.plot(gfiblim, 1632, marker = '*', color = 'blue', markersize = 30, label="This work", linestyle='None', zorder=3)
ax1.xaxis.set_tick_params(labelsize = 29)
ax1.yaxis.set_tick_params(labelsize = 29)
ax1.legend(loc = 'upper left', fontsize = 30)

ax2.plot(maglim, surf_density_actf, color = 'black', lw=4, zorder=1)
ax2.set_xlabel(r'$g_\mathrm{fiber}$ mag limit', fontsize = 32)
ax2.set_ylabel(r'Net surface density yield ($\Sigma_\mathrm{yield}$)', fontsize = 31)
ax2.xaxis.set_tick_params(labelsize = 29)
ax2.yaxis.set_tick_params(labelsize = 29)
ax2.axvline(x=gfiblim,ls= '--',color='purple', lw=4, zorder=2)
ax2.plot(24.21, 428, marker = '*', color = 'grey', markersize = 30, linestyle='None')
ax2.plot(gfiblim, 1372, marker = '*', color = 'blue', markersize = 30, linestyle='None', zorder=3)


plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paper_figures/script_figure_truth/3panel_sumstat.png', dpi=300, bbox_inches='tight')