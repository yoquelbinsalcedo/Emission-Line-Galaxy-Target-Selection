import numpy as np
from matplotlib import pyplot as plt
from astropy.io.misc.hdf5 import read_table_hdf5
import mpl_scatter_density
import matplotlib.colors as colors

# load in crossmatched cat of hsc wide and spec truth
crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)


# quality cuts for reliable spec-z
o2_snr_truth = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR'])  
chi2_truth = combined_cat['DELTACHI2']
o2_chi2_mask_truth = o2_snr_truth > 10**(0.9 - 0.2*np.log10(chi2_truth))
snr_mask_truth = np.logical_or(o2_chi2_mask_truth, chi2_truth > 25)
snr_mask_truth_fail = ~ snr_mask_truth


# grizy colors with reliable spec-z
gr_truth = combined_cat['g_mag'][snr_mask_truth] - combined_cat['r_mag'][snr_mask_truth]
ri_truth = combined_cat['r_mag'][snr_mask_truth] - combined_cat['i_mag'][snr_mask_truth]
iz_truth = combined_cat['i_mag'][snr_mask_truth] - combined_cat['z_mag'][snr_mask_truth]
zy_truth = combined_cat['z_mag'][snr_mask_truth] - combined_cat['y_mag'][snr_mask_truth]
specz_truth = combined_cat['Z'][snr_mask_truth]



fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(24, 12), ncols=1, nrows=4, sharex=True, subplot_kw={'projection': 'scatter_density'})
# fig.suptitle('grizy Colors vs. Spectroscopic Redshift (Density)', fontsize=14)

ax0.scatter_density(specz_truth, gr_truth)
ax0.set_ylabel('g-r', fontsize = 25)
ax0.set_xlim(0, 2)
ax0.set_ylim(-0.5, 1)
ax0.set_aspect('equal', adjustable='box')

ax1.scatter_density(specz_truth, ri_truth)
ax1.set_ylabel('r-i', fontsize = 25)
ax1.set_xlim(0, 2)
ax1.set_ylim(-0.5, 1)
ax1.set_aspect('equal', adjustable='box')

ax2.scatter_density(specz_truth, iz_truth)
ax2.set_ylabel('i-z', fontsize = 25)
ax2.set_xlim(0, 2)
ax2.set_ylim(-0.5, 1)
ax2.set_aspect('equal', adjustable='box')

ax3.scatter_density(specz_truth, zy_truth)
ax3.set_xlim(0, 2)
ax3.set_xlabel('Spectroscopic redshift', fontsize = 20)
ax3.set_ylabel('z-y', fontsize = 25)
ax3.set_ylim(-0.5, 1)
ax3.set_aspect('equal', adjustable='box')


plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paper_figures/script_figure_truth/colors_versus_redshift.png', dpi = 300, bbox_inches='tight')
