import numpy as np
import pandas as pd 
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


# photometry for reliable spec-z
ri_rongpu = combined_cat['r_mag'][snr_mask_rongpu] - combined_cat['i_mag'][snr_mask_rongpu]
iz_rongpu = combined_cat['i_mag'][snr_mask_rongpu] - combined_cat['z_mag'][snr_mask_rongpu]
iy_rongpu = combined_cat['i_mag'][snr_mask_rongpu] - combined_cat['y_mag'][snr_mask_rongpu]
gfiber_rongpu = combined_cat['g_fiber_mag'][snr_mask_rongpu]
specz_rongpu = combined_cat['Z'][snr_mask_rongpu]

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

# colormap to use
cmap = cc.cm.kbc
cmap.set_extremes(under = 'red', over = 'springgreen') 

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(24, 10), 
                                                       #subplot_kw=dict(aspect='equal'),
                                                       #constrained_layout = True,
                                                       )
vmin = 1.1
vmax = 1.6
arrow_color = 'black'

sd0 = ax0.scatter(ri_rongpu, iy_rongpu, c = specz_rongpu, cmap = cmap, s = 6, alpha = 0.1 , vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri_rongpu, iz_rongpu, c = specz_rongpu, cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy_rongpu, iz_rongpu, c = specz_rongpu, cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x = 100, y = 100, c = specz_rongpu[0], cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)

ri_lim = (-0.5, 1.5)
iy_lim = (0, 1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize = 27)
ax0.set_ylabel('i-y', fontsize = 27)
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
x_new = np.arange(0.35 + iyshift , 2, .05)
y_new = x_new - 0.19 + rishift
ax0.plot(y_new, x_new, c = 'black') 
ax0.axhline(y = 0.35 + iyshift , xmax = x_new[0] - 0.05 , c = 'black')  
ax0.xaxis.set_tick_params(labelsize = 16)
ax0.yaxis.set_tick_params(labelsize = 16)
ax0.arrow(-0.35, 0.35 + iyshift, 0, 0.2, head_width=0.05, color=arrow_color)
ax0.arrow(0.77, 0.94 , -0.2, 0, head_width=0.05, color=arrow_color)

ax1.set_xlabel('r-i', fontsize = 27)
ax1.set_ylabel('i-z', fontsize = 27)
ax1.xaxis.set_tick_params(labelsize = 16)
ax1.yaxis.set_tick_params(labelsize = 16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)
ax1.axhline(y = izmin , color='black')
ax1.arrow(0.40, izmin, 0, 0.2, head_width=0.05, color=arrow_color)

ax2.set_xlabel('i-y', fontsize = 27)
ax2.set_ylabel('i-z', fontsize = 27)
ax2.xaxis.set_tick_params(labelsize = 16)
ax2.yaxis.set_tick_params(labelsize = 16)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.axhline(y = izmin, xmin=0.35 - iyshift + 0.02,  color='black')
ax2.axvline(x = 0.35 + iyshift , color='black')
ax2.arrow(0.35 + iyshift, 0.70, 0.2, 0, head_width=0.04, color=arrow_color)
ax2.arrow(0.75, izmin, 0, 0.2, head_width=0.04, color=arrow_color)

cbar = plt.colorbar(sdummy, ax=[ax0, ax1, ax2], orientation='horizontal',  extend = 'both', shrink = 0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Spectroscopic redshift', fontsize = 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/3panel_color_color_selection.png', dpi = 300, bbox_inches='tight')
