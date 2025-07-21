
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5


# load in crossmatched cat of hsc wide and spec truth
crossmatch_file_path =  '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)


# quality cuts for reliable spec-z
o2_snr_rongpu = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR'])  
chi2_rongpu = combined_cat['DELTACHI2']
o2_chi2_mask_rongpu = o2_snr_rongpu > 10**(0.9 - 0.2*np.log10(chi2_rongpu))
snr_mask_rongpu = np.logical_or(o2_chi2_mask_rongpu, chi2_rongpu > 25)
snr_mask_rongpu_fail = ~ snr_mask_rongpu


# photometry for reliable spec-z
ri_rongpu = combined_cat['r_mag'][snr_mask_rongpu] - combined_cat['i_mag'][snr_mask_rongpu]
iz_rongpu = combined_cat['i_mag'][snr_mask_rongpu] - combined_cat['z_mag'][snr_mask_rongpu]
iy_rongpu = combined_cat['i_mag'][snr_mask_rongpu] - combined_cat['y_mag'][snr_mask_rongpu]
gfiber_rongpu = combined_cat['g_fiber_mag'][snr_mask_rongpu]

specz_rongpu = combined_cat['Z'][snr_mask_rongpu]
clean_specz_number = len(specz_rongpu)

print(f'Before applying exposure & good spec-z cuts, we have {len(combined_cat)}. After, we have {clean_specz_number} elgs')

# best case for 1.1 < z <  1.6 for rioffset, iyoffset, izmin, and gfiblim values
optimization_csv_path = str(Path('../data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'))
opt_params = pd.read_csv(optimization_csv_path)


rishift, iyshift, izmin, gfiblim = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]


# color and g-fiber cuts
target_iyri_cut = -0.19
target_iy_cut = 0.35
color_mask_iyri = np.logical_and(
    (ri_rongpu < iy_rongpu + target_iyri_cut + rishift),
    (iy_rongpu > target_iy_cut + iyshift))
color_mask_iz = (iz_rongpu) > izmin
gfiber_mask = gfiber_rongpu < gfiblim
ccuts = np.logical_and.reduce((color_mask_iyri, color_mask_iz, gfiber_mask))

fig, ax = plt.subplots(figsize=(12,8))
ax.hist(specz_rongpu[ccuts], bins = np.linspace(0, 2, 100), histtype='step', color='black', label='Spectroscopic Truth DESI-2 ELGs')
ax.axvline(x=1.1,ls='--', color='black')
ax.axvline(x=1.6,ls='--', color='black')
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
ax.set_xlabel('Spec z', fontsize=22)
ax.set_ylabel('Count', fontsize=22)
ax.legend(fontsize=18, loc = 'best')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/zhist_final_cuts_truth_1-1_1-6.png', dpi=300, bbox_inches='tight')