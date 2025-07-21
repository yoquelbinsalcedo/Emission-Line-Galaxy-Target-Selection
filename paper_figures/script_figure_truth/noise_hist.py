import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5
import pandas as pd


crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/noisy_spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)

# quality cuts for reliable spec-z for spectroscopic truth sample
o2_snr_rongpu = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR'])   
chi2_rongpu = combined_cat['DELTACHI2']
o2_chi2_mask_rongpu = o2_snr_rongpu > 10**(0.9 - 0.2*np.log10(chi2_rongpu))
snr_mask_rongpu = np.logical_or(o2_chi2_mask_rongpu, chi2_rongpu > 25)
snr_mask_rongpu_fail =~ snr_mask_rongpu

# noisy colors
ri_rongpu = combined_cat['r_mag_noisy'][snr_mask_rongpu] - combined_cat['i_mag_noisy'][snr_mask_rongpu]
iz_rongpu = combined_cat['i_mag_noisy'][snr_mask_rongpu] - combined_cat['z_mag_noisy'][snr_mask_rongpu]
iy_rongpu = combined_cat['i_mag_noisy'][snr_mask_rongpu] - combined_cat['y_mag_noisy'][snr_mask_rongpu]
gfibermag_rongpu = combined_cat['g_fiber_mag'][snr_mask_rongpu]

specz_rongpu = combined_cat['Z'][snr_mask_rongpu]


# best case for 1.1 < z < 1.6 case with noisy photometry
iz_noise_cut = 0.35
izri_noise_cut = 0.05
iy_noise_cut = 0.50
optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/noisy_spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path)
rishift_final, iyshift_final, izmin_final, gfiblim_final = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]


# final selection cuts using optimization results
color_mask_rongpu_izri = np.logical_and((ri_rongpu < iz_rongpu + izri_noise_cut + rishift_final),
                             (iy_rongpu > iy_noise_cut + iyshift_final))
color_mask_rongpu_iz = iz_rongpu > izmin_final
color_cuts_rongpu = np.logical_and(color_mask_rongpu_izri, color_mask_rongpu_iz) 

color_cuts_final = np.logical_and(color_cuts_rongpu, gfibermag_rongpu < gfiblim_final)


# plot the final distribution for the best selection with noisy y-band

fig, ax = plt.subplots(figsize=(12,8))
ax.hist(specz_rongpu[color_cuts_final], bins = np.linspace(0, 2, 100), histtype='step', color='black')
ax.axvline(x = 1.1,ls= '--', color='black')
ax.axvline(x = 1.6,ls= '--', color='black')
ax.xaxis.set_tick_params(labelsize = 18)
ax.yaxis.set_tick_params(labelsize = 18)
ax.set_xlabel('Spec z', fontsize = 25)
ax.set_ylabel('Count', fontsize = 25)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/noise_hist', dpi = 300, bbox_inches='tight')
