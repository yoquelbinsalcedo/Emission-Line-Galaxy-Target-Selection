import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import colorcet as cc
from astropy.io.misc.hdf5 import read_table_hdf5


def complete_purity_cutsonly(colorcuts, specz, zcut=[1.1, 1.6]):

    mask_zrange = np.logical_and(specz > zcut[0], specz < zcut[1])
    completeness = np.sum(np.logical_and(colorcuts, mask_zrange)) / np.sum(mask_zrange)
    purity = np.sum(np.logical_and(colorcuts, mask_zrange)) / np.sum(colorcuts)

    return completeness, purity


path_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/noisy_spec_truth_hsc_wide_crossmatch.hdf5'
cat = read_table_hdf5(input=path_cat)

# colors to plot and noisy y-band
specz = cat['Z']
iz = cat['i_mag'] - cat['z_mag']
ri = cat['r_mag'] - cat['i_mag']
iy = cat['i_mag'] - cat['y_mag']


iz_noisy = cat['i_mag_noisy'] - cat['z_mag_noisy']
ri_noisy = cat['r_mag_noisy'] - cat['i_mag_noisy']
iy_noisy = cat['i_mag_noisy'] - cat['y_mag_noisy']

# best cuts for 1.1 < z < 1.6 case
iyri_cut = 0.19
iy_cut = 0.35
optimization_csv_path_clean = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path_clean)
rishift, iyshift, izmin, gfiblim = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]
color_cuts_clean = np.logical_and.reduce((ri < iy - iyri_cut + rishift, iy > iy_cut + iyshift, iz > izmin))


# best case for 1.1 < z < 1.6 case with noisy photometry
iz_noise_cut = 0.35
izri_noise_cut = 0.05
iy_noise_cut = 0.50
optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/noisy_spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path)
rishift_noisy, iyshift_noisy, izmin_noisy, gfiblim_noisy = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]


# noisy selection cuts using optimization results
color_mask_rongpu_izri = np.logical_and((ri_noisy < iz_noisy + izri_noise_cut + rishift_noisy),
                             (iy_noisy > iy_noise_cut + iyshift_noisy))
color_mask_rongpu_iz = iz_noisy > izmin_noisy
color_cuts_noisy_new = np.logical_and(color_mask_rongpu_izri, color_mask_rongpu_iz)



clean_comp, clean_pure = complete_purity_cutsonly(color_cuts_clean, specz)
print(f'color cuts used for noise optimal selection with no degraded photometry: comp = {clean_comp:.2}, pure = {clean_pure:.2}')
noise_new_comp, noise_new_pure = complete_purity_cutsonly(color_cuts_noisy_new, specz)
print(f'ri_noisy < iz_noisy + {izri_noise_cut + rishift_noisy :.2} & iz_noisy > {iz_noise_cut:.2} & iy_noisy > {iy_noise_cut + iyshift_noisy:.2}: completeness = {noise_new_comp:.2}, purity = {noise_new_pure:.2}')

# colormap to use
cmap = cc.cm.kbc
cmap.set_extremes(under = 'red', over = 'springgreen') 
vmin = 1.1
vmax = 1.6

fig, ([ax0, ax1, ax2], [ax3, ax4, ax5]) = plt.subplots(nrows=2, ncols=3, figsize=(12,8),
                                                       # subplot_kw=dict(aspect='equal'),
                                                       constrained_layout = True,
                                                       )

sd0 = ax0.scatter(ri[color_cuts_clean], iy[color_cuts_clean], c = specz[color_cuts_clean] , cmap = cmap, s=3, alpha =.1, vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri[color_cuts_clean], iz[color_cuts_clean], c = specz[color_cuts_clean], cmap = cmap, s=3, alpha =.1, vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy[color_cuts_clean], iz[color_cuts_clean], c = specz[color_cuts_clean], cmap =  cmap, s=3, alpha =.1, vmin=vmin, vmax=vmax)

sd3 = ax3.scatter(ri_noisy[color_cuts_noisy_new], iy_noisy[color_cuts_noisy_new], c = specz[color_cuts_noisy_new] , cmap = cmap, s=3, alpha=.1, vmin=vmin, vmax=vmax)
sd4 = ax4.scatter(ri_noisy[color_cuts_noisy_new], iz_noisy[color_cuts_noisy_new], c = specz[color_cuts_noisy_new], cmap = cmap, s=3, alpha=.1, vmin=vmin, vmax=vmax)
sd5 = ax5.scatter(iy_noisy[color_cuts_noisy_new], iz_noisy[color_cuts_noisy_new], c = specz[color_cuts_noisy_new] , cmap = cmap, s=3, alpha=.1, vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x = 100, y = 100, c=specz[0], cmap = cmap, s = 3, alpha=1 , vmin=vmin, vmax=vmax)

noise = 'noise'
# after color cuts
ri_lim = (-0.5,1.5)
iy_lim = (0,1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize = 27)
ax0.set_ylabel('i-y', fontsize = 27)
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
ax0.xaxis.set_tick_params(labelsize = 16)
ax0.yaxis.set_tick_params(labelsize = 16)
ax0.axhline(y = iy_cut + iyshift, xmax = 0.380, ls = '--', c = 'black') 
x = np.arange(iy_cut + iyshift, 2, .05)
y = x - iyri_cut + 0.04
ax0.plot(y ,x , ls = '--', c = 'black')

ax1.set_xlabel('r-i', fontsize = 27)
ax1.set_ylabel('i-z', fontsize = 27)
ax1.xaxis.set_tick_params(labelsize = 16)
ax1.yaxis.set_tick_params(labelsize = 16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)
ax1.set_title(f'Clean color cuts: Completeness = {clean_comp:.2}, Purity= {clean_pure:.2}', fontsize = 15, pad=10)
ax1.axhline(y = izmin, ls = '--', c = 'black') 

ax2.set_xlabel('i-y', fontsize = 27)
ax2.set_ylabel('i-z', fontsize = 27)
ax2.xaxis.set_tick_params(labelsize = 16)
ax2.yaxis.set_tick_params(labelsize = 16)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.axhline(y = izmin , color='black', ls = '--')
ax2.axvline(x = iy_cut + iyshift, ymin = iz_noise_cut + 0.180, ls = '--', color='black')


# colors cut and plotted with noisy photometry
ax3.set_xlabel('r-i', fontsize = 27)
ax3.set_ylabel('i-y', fontsize = 27)
ax3.set_xlim(*ri_lim)
ax3.set_ylim(*iy_lim)
ax3.xaxis.set_tick_params(labelsize = 16)
ax3.yaxis.set_tick_params(labelsize = 16)
ax3.axhline(y = iy_noise_cut + iyshift_noisy, ls = '--', c = 'black') 

ax4.set_title(f'Noisy color cuts: Completeness = {noise_new_comp:.2}, Purity = {noise_new_pure:.2}', fontsize = 15, pad=10)
ax4.set_xlabel('r-i', fontsize = 27)
ax4.set_ylabel('i-z', fontsize = 27)
ax4.xaxis.set_tick_params(labelsize = 16)
ax4.yaxis.set_tick_params(labelsize = 16)
ax4.set_xlim(*ri_lim)
ax4.set_ylim(*iz_lim)
ax4.axhline(y = iz_noise_cut, xmax = 0.45, ls = '--', c = 'black')
x_noise = np.arange(iz_noise_cut, 2, .05)
y_noise = x_noise + izri_noise_cut
ax4.plot(y_noise , x_noise , ls = '--', c = 'black')


ax5.set_xlabel('i-y', fontsize = 27)
ax5.set_ylabel('i-z', fontsize = 27)
ax5.xaxis.set_tick_params(labelsize = 16)
ax5.yaxis.set_tick_params(labelsize = 16)
ax5.set_xlim(*iy_lim)
ax5.set_ylim(*iz_lim)
ax5.set_xlim(*iy_lim)
ax5.set_ylim(*iz_lim)
ax5.axhline(y = iz_noise_cut , color='black', ls = '--')
ax5.axvline(x = iy_noise_cut + iyshift_noisy, ymin = iz_noise_cut + 0.10, ls = '--', color='black')


cbar = plt.colorbar(sdummy, ax=[ax3, ax4, ax5], orientation='horizontal',  extend ='both', shrink=0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Spectroscopic redshift', fontsize = 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paper_figures/script_figure_truth/noise_colorcolor_cut.png', dpi=300, bbox_inches='tight')