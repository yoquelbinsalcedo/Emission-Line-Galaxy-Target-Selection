import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

# load in spectroscopic truth sample
truth_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth.hdf5'
truth_cat = read_table_hdf5(input=truth_file_path)

# criteria for good redshifts
o2_snr = truth_cat['OII_FLUX']*np.sqrt(truth_cat['OII_FLUX_IVAR'])   
chi2 = truth_cat['DELTACHI2']
snr_mask1 = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
snr_mask = np.logical_or(snr_mask1, chi2 > 25)
x = np.linspace(-1, 5, 100)
y = (0.9 - 0.2*(x))


# colors for color coding
cmap = cc.cm.kbc
cmap.set_extremes(under = 'red', over = 'springgreen')


# make scatter plot of objects that pass and fail snr_mask
fig, ax = plt.subplots()
ax.scatter(np.log10(chi2), np.log10(o2_snr), c = truth_cat['Z'], cmap = cmap, s = 2, alpha = 0.1, vmin= 1.1, vmax= 1.6)
ax.plot(x, y, color = 'black', label = 'y = 0.9 - 0.2 * x')
cdummy = ax.scatter(x = 100, y = 100, c = truth_cat['Z'][0], cmap = cmap, s = 5, alpha = 1 ,vmin= 1.1, vmax= 1.6)
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 2)
ax.set_xlabel('log($\Delta\chi^2$)', fontsize = 30)
ax.set_ylabel('log(S/N [OII])', fontsize = 30)
ax.xaxis.set_tick_params(labelsize = 22)
ax.yaxis.set_tick_params(labelsize = 22)
ax.axvline(1.40, ls = '--', label = '$\Delta\chi^2$ = 25', c = 'black')
cbar = plt.colorbar(cdummy, extend = 'both')
cbar.set_label('Spectroscopic redshift', fontsize = 30)
cbar.ax.tick_params(labelsize=20)
ax.legend(fontsize = 22)
xfill = np.linspace(-1, 1.40,2)
yfill1 = np.linspace(1.1, 0.6200000000000001,2)
yfill2 = np.linspace(-1,-1, 2)
ax.fill_between(xfill, yfill1, yfill2, color = 'grey', alpha = 0.5)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/scatter_good_specz_cuts.png', dpi=300, bbox_inches='tight')


unique_elgs_number = len(truth_cat)
reliable_z_number = np.sum(snr_mask)
frac_good_z = reliable_z_number / unique_elgs_number
print(f'The number of unique ELGs is {unique_elgs_number}')
print(f'The number of ELGs with a reliable spec-z is {reliable_z_number}')
print(f'The fraction with good redshifts are {frac_good_z}')
