import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

# load in crossmatched cat of hsc wide and spec truth
crossmatch_input_name = 'spec_truth_hsc_wide_crossmatch'
crossmatch_input = f'{crossmatch_input_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_input}'))
combined_cat = read_table_hdf5(input=crossmatch_file_path)

# quality cuts for reliable spec-z
o2_snr_rongpu = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR'])  
chi2_rongpu = combined_cat['DELTACHI2']
o2_chi2_mask_rongpu = o2_snr_rongpu > 10**(0.9 - 0.2*np.log10(chi2_rongpu))
snr_mask_rongpu = np.logical_or(o2_chi2_mask_rongpu, chi2_rongpu > 25)
snr_mask_rongpu_fail = ~ snr_mask_rongpu

# HSC wide photometry for reliable spec-z
gr_wide = combined_cat['g_mag'][snr_mask_rongpu] - combined_cat['r_mag'][snr_mask_rongpu]
rz_wide = combined_cat['r_mag'][snr_mask_rongpu] - combined_cat['z_mag'][snr_mask_rongpu]
gz_wide = combined_cat['g_mag'][snr_mask_rongpu] - combined_cat['z_mag'][snr_mask_rongpu]

gr_deep = combined_cat['gmag'][snr_mask_rongpu] - combined_cat['rmag'][snr_mask_rongpu]
gz_deep = combined_cat['gmag'][snr_mask_rongpu] - combined_cat['zmag'][snr_mask_rongpu]
rz_deep = combined_cat['rmag'][snr_mask_rongpu] - combined_cat['zmag'][snr_mask_rongpu]

specz_rongpu = combined_cat['Z'][snr_mask_rongpu]

gr_lim_clean = (-0.5, 1)
rz_lim_clean = (-0.5, 2)

# DESI ELG color cuts
x_slope_plus = np.linspace(0.15, 0.705, 100)
y_slope_plus = 0.5 * x_slope_plus + 0.1

x_slope_minus = np.linspace(0.705, 1.5, 100)
y_slope_minus = -1.2 * x_slope_minus + 1.3

# Truth sample selection cuts
# HSC deep selection:
# 19.5 < gfibermag < 24.55
# g - r < 0.8
# g - r < 1.22 - 0.7 * (r-z)

truth_neg_x = np.linspace(0.60, 2.0, 100)
truth_neg_y = 1.22 - 0.7*truth_neg_x

# plot g-r vs r-z using both deep and wide imaging from HSC side by side

vmin_specz = 1.1
vmax_specz = 1.6

# colormap to use
cmap = cc.cm.kbc 
cmap.set_extremes(under = 'red', over = 'springgreen') 


fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))


sd0 = ax0.scatter(rz_deep, gr_deep, c=specz_rongpu, cmap=cmap, s=12, alpha=0.1, vmin=vmin_specz, vmax=vmax_specz)
sd1 = ax1.scatter(rz_wide, gr_wide, c=specz_rongpu, cmap=cmap, s=12, alpha=0.1, vmin=vmin_specz, vmax=vmax_specz)
sdummy = ax0.scatter(x=100, y=100, c=specz_rongpu[0], cmap=cmap, s=12, alpha=1, vmin=vmin_specz, vmax=vmax_specz)

ax0.set_title('HSC Deep', fontsize=29)
ax0.set_xlabel('r-z', fontsize=27)
ax0.set_ylabel('g-r', fontsize=27)
ax0.xaxis.set_tick_params(labelsize=16)
ax0.yaxis.set_tick_params(labelsize=16)
ax0.set_xlim(*rz_lim_clean)
ax0.set_ylim(*gr_lim_clean)

# draw DESI1 ELG selection cuts 
ax0.plot([0.15,0.15], [-10, y_slope_plus[0]], ls='--', color='grey')
ax0.plot(x_slope_plus, y_slope_plus, ls='--', color='grey', label='DESI LOP')
ax0.plot(x_slope_minus, y_slope_minus, ls='--', color='grey')

# draw truth selection cuts
ax0.plot([-0.5, truth_neg_x[0]], [0.8, 0.8], ls='-', color='black')
ax0.plot(truth_neg_x, truth_neg_y, ls='-', color='black')

ax1.set_title('HSC Wide', fontsize=29)
ax1.set_xlabel('r-z', fontsize=27)
ax1.set_ylabel('g-r', fontsize=27)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)
ax1.set_xlim(*rz_lim_clean)
ax1.set_ylim(*gr_lim_clean)

# draw DESI1 ELG selection cuts 
ax1.plot([0.15,0.15], [-10, y_slope_plus[0]], ls='--', color='grey')
ax1.plot(x_slope_plus, y_slope_plus, ls='--', color='grey', label='DESI LOP')
ax1.plot(x_slope_minus, y_slope_minus, ls='--', color='grey')

# draw truth sample selection cuts
ax1.plot([-0.5, truth_neg_x[0]], [0.8, 0.8], ls='-', color='black')
ax1.plot(truth_neg_x, truth_neg_y, ls='-', color='black')

cbar = plt.colorbar(sdummy, ax=[ax0, ax1], orientation='horizontal', extend='both', shrink=0.80)
cbar.ax.tick_params(labelsize=16)
cbar.set_label(f'Spectroscopic redshift', fontsize=27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/color_color_hsc_wide_vs_deep.png', dpi=300, bbox_inches='tight')
