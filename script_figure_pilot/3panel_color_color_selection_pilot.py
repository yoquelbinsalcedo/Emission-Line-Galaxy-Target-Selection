import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

crossmatch_input_name = 'desi2_pilot_hsc_wide_crossmatch'
crossmatch_input = f'{crossmatch_input_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_input}'))
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

# specz combined catalog
specz = combined_cat['Z'][snr_mask_comb]

# best cuts for 1.1 < z < 1.6 case 
rishift, iyshift, izmin, gfiblim = 0, 5.979861E-02, 3.634577E-01, 2.422445E+01


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

sd0 = ax0.scatter(ri, iy, c = specz , cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri, iz, c = specz, cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy, iz, c = specz , cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x = 100, y = 100, c = combined_cat['Z'][0], cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)

# #after color cuts
ri_lim = (-0.5,1.5)
iy_lim = (0,1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize = 27)
ax0.set_ylabel('i-y', fontsize = 27)
ax0.axhline(y = 0.35, xmax = 0.33, ls = '--', c = 'black') 
ax0.axhline(y = 0.35 + iyshift , xmax = 0.358 , c = 'black')  
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
x = np.arange(0.35, 2, .05)
y = x - 0.19
ax0.plot(y ,x , ls = '--', c = 'black')
x_new = np.arange(0.41 , 2, .05)
y_new = x_new - 0.19
ax0.plot(y_new ,x_new, c = 'black') 
ax0.xaxis.set_tick_params(labelsize = 16)
ax0.yaxis.set_tick_params(labelsize = 16)
ax0.arrow(-0.35, 0.35 + iyshift, 0, 0.2, head_width=0.05, color=arrow_color)
ax0.arrow(0.75, 0.94 , -0.2, 0, head_width=0.05, color=arrow_color)

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
ax2.axhline(y = izmin , color='black')
ax2.axvline(x = 0.35, ls = '--', color='black')
ax2.axvline(x = 0.35 + iyshift , color='black')
ax2.arrow(0.35 + iyshift, 0.70, 0.2, 0, head_width=0.04, color=arrow_color)
ax2.arrow(0.70, izmin, 0, 0.2, head_width=0.04, color=arrow_color)

cbar = plt.colorbar(sdummy, ax=[ax0, ax1, ax2], orientation='horizontal',  extend = 'both', shrink = 0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Spectroscopic redshift', fontsize = 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_pilot/3panel_color_color_selection_pilot.png', dpi = 300, bbox_inches='tight')
