import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import units as u
from astropy import table
from astropy.table import hstack
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.coordinates import SkyCoord
import colorcet as cc


crossmatch_input_name = 'spec_truth_hsc_wide_crossmatch'
crossmatch_input = f'{crossmatch_input_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_input}'))
hscw_elgs_rongpu = read_table_hdf5(input=crossmatch_file_path)


# load in cosmos2020 catalog
dir_in_cosmos = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/input/cosmos2020_farmer.fits'
cat = table.Table.read(dir_in_cosmos, format='fits', hdu=1)
colormaskx = np.logical_and.reduce((np.isfinite(cat['photoz']),
                                    np.isfinite(cat['CFHT_u_MAG']),
                                    np.isfinite(cat['HSC_g_MAG']),
                                    np.isfinite(cat['HSC_r_MAG']),
                                    np.isfinite(cat['HSC_i_MAG']),
                                    np.isfinite(cat['HSC_z_MAG']),
                                    np.isfinite(cat['HSC_y_MAG']),
                                    (np.logical_or(cat['HSC_g_MAG'] < 24.5, cat['HSC_r_MAG'] < 24.5))))

cat_cosmos = cat[colormaskx]

# combine the hsc/elg catalog with cosmos2020
ra_cat_hsc = hscw_elgs_rongpu['TARGET_RA']
dec_cat_hsc = hscw_elgs_rongpu['TARGET_DEC']
ra_cos = cat_cosmos['RA']
dec_cos = cat_cosmos['DEC']
cat_hsc_coord = SkyCoord(ra_cat_hsc*u.degree, dec_cat_hsc*u.degree)
cos_coord = SkyCoord(ra_cos, dec_cos)
idx_cos_hsc, d2d_cos_hsc, d3d_cos_hsc = cat_hsc_coord.match_to_catalog_sky(cos_coord)
dmask_cos_hsc = d2d_cos_hsc.arcsec < 1
combined_cat_hsc_cos = hstack([hscw_elgs_rongpu, cat_cosmos[idx_cos_hsc]])[dmask_cos_hsc]
print(f'the number of objects in this doubly cross-matched cat is {len(combined_cat_hsc_cos)}')

# masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat_hsc_cos['OII_FLUX']*np.sqrt(combined_cat_hsc_cos['OII_FLUX_IVAR'])   
chi2_comb = combined_cat_hsc_cos['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail = ~snr_mask_comb

# colors for objects that did NOT pass the above snr/chi2 cuts
ri = combined_cat_hsc_cos['r_mag'][snr_mask_comb_fail] - combined_cat_hsc_cos['i_mag'][snr_mask_comb_fail]
iz = combined_cat_hsc_cos['i_mag'][snr_mask_comb_fail] - combined_cat_hsc_cos['z_mag'][snr_mask_comb_fail]
iy = combined_cat_hsc_cos['i_mag'][snr_mask_comb_fail] - combined_cat_hsc_cos['y_mag'][snr_mask_comb_fail]

photoz = combined_cat_hsc_cos['photoz'][snr_mask_comb_fail]

# best case for 1.1 < z <  1.6 for rioffset, iyoffset, izmin, and gfiblim values
optimization_csv_path = str(Path('../data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'))
opt_params = pd.read_csv(optimization_csv_path)
rishift, iyshift, izmin, gfiblim = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]

# color and g-fiber cuts
target_iyri_cut = -0.19
target_iy_cut = 0.35

# colormap to use
cmap = cc.cm.kbc
cmap.set_extremes(under = 'red', over = 'springgreen') 
vmin = 1.1
vmax = 1.6
arrow_color = 'black'

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize = (24,10)
                                                       #subplot_kw=dict(aspect='equal'),
                                                       #constrained_layout = True,
                                                       )

sd0 = ax0.scatter(ri, iy, c = photoz, cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri, iz, c = photoz, cmap = cmap, s = 12, alpha = 1, vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy, iz, c = photoz, cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x = 100, y = 100, c = photoz[0], cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)

# #after color cuts
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
cbar.set_label('Lephare photoz', fontsize= 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/color_color_fail.png', dpi=300, bbox_inches='tight' )
