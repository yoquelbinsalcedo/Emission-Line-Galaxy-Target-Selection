import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from astropy.io.misc.hdf5 import read_table_hdf5
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


def complete_purity_cutsonly(colorcuts, photozs, zcut=[1.1, 1.6]):

    zrange = np.logical_and(photozs > zcut[0], photozs < zcut[1])
    completeness = np.sum(np.logical_and(colorcuts, zrange)) / np.sum(zrange)
    purity = np.sum(np.logical_and(colorcuts, zrange)) / np.sum(colorcuts)

    return completeness, purity


input_name = 'noisy_cosmos2020'
input = f'{input_name}.hdf5'
output_file_path = str(Path(f'../data/processed/hdf5/{input}'))
cat = read_table_hdf5(input=output_file_path).to_pandas()

# train a new RF using the noisy y-band

X = cat[['iy_noisy', 'ri_noisy', 'iz_noisy']]
photz = cat['photoz']
y = np.logical_and(photz > 1.1, photz < 1.6)

X_train, X_test, photz_train, photz_test, y_train, y_test = train_test_split(X, photz, y, test_size=0.20)

# training the RF
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predict_prob = clf.predict_proba(cat[['iy_noisy', 'ri_noisy', 'iz_noisy']])[:, 1]

# colors to plot and noisey y-band
photoz = cat['photoz']
iz = cat['iz']
ri = cat['ri']
iy = cat['iy']


iz_noisy = cat['iz_noisy']
ri_noisy = cat['ri_noisy']
iy_noisy = cat['iy_noisy']

# best cuts for 1.1 < z < 1.6 case
iyri_cut = 0.19
iy_cut = 0.35
rishift, iyshift, izmin, gfiblim = 0, 5.979861E-02, 3.634577E-01, 2.422445E+01
color_cuts_clean = np.logical_and.reduce((ri < iy - iyri_cut + rishift, iy > iy_cut + iyshift, iz > izmin))


# new best color cuts using the noisy photometry
iz_noise_cut = 0.35
izri_noise_cut = 0.05
iy_noise_cut = 0.50
rishift_noisy, iyshift_noisy, izmin_noisy, gfiblim_noisy = -2.666195E-03, 5.321480E-02, 3.528028E-01, 2.425522E+01

# noisy selection cuts using optimization results
color_mask_rongpu_izri = np.logical_and((ri_noisy < iz_noisy + izri_noise_cut + rishift_noisy),
                             (iy_noisy > iy_noise_cut + iyshift_noisy))
color_mask_rongpu_iz = iz_noisy > izmin_noisy
color_cuts_noisy_new = np.logical_and(color_mask_rongpu_izri, color_mask_rongpu_iz)



clean_comp, clean_pure = complete_purity_cutsonly(color_cuts_clean, photoz)
print(f'color cuts used for noise optimal selection with no degraded y band: comp = {clean_comp:.2}, pure = {clean_pure:.2}')
noise_new_comp, noise_new_pure = complete_purity_cutsonly(color_cuts_noisy_new, photoz)
print(f'ri_noisy < iz_noisy + {izri_noise_cut + rishift_noisy :.2} & iz_noisy > {iz_noise_cut:.2} & iy_noisy > {iy_noise_cut + iyshift_noisy:.2}: completeness = {noise_new_comp:.2}, purity = {noise_new_pure:.2}')

fig, ([ax0, ax1, ax2], [ax3, ax4, ax5]) = plt.subplots(nrows=2, ncols=3, figsize=(12,8),
                                                       # subplot_kw=dict(aspect='equal'),
                                                       constrained_layout = True,
                                                       )

sd0 = ax0.scatter(ri[color_cuts_clean], iy[color_cuts_clean], c = predict_prob[color_cuts_clean] , cmap ='viridis', s=3, alpha =.1, vmin=0, vmax=1)
sd1 = ax1.scatter(ri[color_cuts_clean], iz[color_cuts_clean], c = predict_prob[color_cuts_clean], cmap = 'viridis', s=3, alpha =.1, vmin=0, vmax=1)
sd2 = ax2.scatter(iy[color_cuts_clean], iz[color_cuts_clean], c = predict_prob[color_cuts_clean], cmap = 'viridis', s=3, alpha =.1, vmin=0, vmax=1)

sd3 = ax3.scatter(ri_noisy[color_cuts_noisy_new], iy_noisy[color_cuts_noisy_new], c = predict_prob[color_cuts_noisy_new] , cmap = 'viridis', s=3, alpha=.1, vmin=0, vmax=1)
sd4 = ax4.scatter(ri_noisy[color_cuts_noisy_new], iz_noisy[color_cuts_noisy_new], c = predict_prob[color_cuts_noisy_new], cmap = 'viridis', s=3, alpha=.1, vmin=0, vmax=1)
sd5 = ax5.scatter(iy_noisy[color_cuts_noisy_new], iz_noisy[color_cuts_noisy_new], c = predict_prob[color_cuts_noisy_new] , cmap='viridis', s=3, alpha=.1, vmin=0, vmax=1)
sdummy = ax1.scatter(x = 100, y = 100, c=predict_prob[0], cmap = 'viridis', s = 3, alpha=1 ,vmin=0, vmax=1)

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
ax0.axhline(y = iy_cut + iyshift, xmax = 0.340, ls = '--', c = 'black') 
x = np.arange(iy_cut + iyshift, 2, .05)
y = x - iyri_cut
ax0.plot(y ,x , ls = '--', c = 'black')

ax1.set_xlabel('r-i', fontsize = 27)
ax1.set_ylabel('i-z', fontsize = 27)
ax1.xaxis.set_tick_params(labelsize = 16)
ax1.yaxis.set_tick_params(labelsize = 16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)
ax1.set_title(f'Clean color cuts: comp = {clean_comp:.2}, pure = {clean_pure:.2}', fontsize = 15)
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
ax2.axvline(x = iy_cut + iyshift, ymin = iz_noise_cut + 0.10, ls = '--', color='black')


# colors cut and plotted with noisy photometry
ax3.set_xlabel('r-i', fontsize = 27)
ax3.set_ylabel('i-y', fontsize = 27)
ax3.set_xlim(*ri_lim)
ax3.set_ylim(*iy_lim)
ax3.xaxis.set_tick_params(labelsize = 16)
ax3.yaxis.set_tick_params(labelsize = 16)
ax3.axhline(y = iy_noise_cut + iyshift_noisy, ls = '--', c = 'black') 

ax4.set_title(f'Noisy color cuts: comp= {noise_new_comp:.2}, pure = {noise_new_pure:.2}', fontsize = 15)
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
cbar.set_label('RF prob', fontsize = 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/noise_colorcolor_cut.png', dpi=300, bbox_inches='tight')