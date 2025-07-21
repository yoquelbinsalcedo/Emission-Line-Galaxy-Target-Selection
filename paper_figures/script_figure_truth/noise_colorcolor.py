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


output_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/noisy_cosmos2020.hdf5'
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

# colors to plot
photoz = cat['photoz']
iz = cat['iz']
ri = cat['ri']
iy = cat['iy']

iz_noisy = cat['iz_noisy']
ri_noisy = cat['ri_noisy']
iy_noisy = cat['iy_noisy']



fig, ([ax0, ax1, ax2], [ax3, ax4, ax5]) = plt.subplots(nrows=2, ncols=3, figsize=(12,8),
                                                       # subplot_kw=dict(aspect='equal'),
                                                       constrained_layout = True,
                                                       )

sd0 = ax0.scatter(ri, iy, c = predict_prob , cmap ='viridis', s=3, alpha =.1, vmin=0, vmax=1)
sd1 = ax1.scatter(ri, iz, c = predict_prob, cmap = 'viridis', s=3, alpha =.1, vmin=0, vmax=1)
sd2 = ax2.scatter(iy, iz, c = predict_prob, cmap = 'viridis', s=3, alpha =.1, vmin=0, vmax=1)

sd3 = ax3.scatter(ri_noisy, iy_noisy, c = predict_prob , cmap = 'viridis', s=3, alpha=.1, vmin=0, vmax=1)
sd4 = ax4.scatter(ri_noisy, iz_noisy, c = predict_prob, cmap = 'viridis', s=3, alpha=.1, vmin=0, vmax=1)
sd5 = ax5.scatter(iy_noisy, iz_noisy, c = predict_prob , cmap='viridis', s=3, alpha=.1, vmin=0, vmax=1)
sdummy = ax1.scatter(x = 100, y = 100, c=predict_prob[0], cmap = 'viridis', s = 3, alpha=1 ,vmin=0, vmax=1)

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

ax1.set_xlabel('r-i', fontsize = 27)
ax1.set_ylabel('i-z', fontsize = 27)
ax1.xaxis.set_tick_params(labelsize = 16)
ax1.yaxis.set_tick_params(labelsize = 16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)
ax1.set_title('Clean')

ax2.set_xlabel('i-y', fontsize = 27)
ax2.set_ylabel('i-z', fontsize = 27)
ax2.xaxis.set_tick_params(labelsize = 16)
ax2.yaxis.set_tick_params(labelsize = 16)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)


# colors cut and plotted with noisy photometry
ax3.set_xlabel('r-i', fontsize = 27)
ax3.set_ylabel('i-y', fontsize = 27)
ax3.set_xlim(*ri_lim)
ax3.set_ylim(*iy_lim)
ax3.xaxis.set_tick_params(labelsize = 16)
ax3.yaxis.set_tick_params(labelsize = 16)

ax4.set_title('Noisy')
ax4.set_xlabel('r-i', fontsize = 27)
ax4.set_ylabel('i-z', fontsize = 27)
ax4.xaxis.set_tick_params(labelsize = 16)
ax4.yaxis.set_tick_params(labelsize = 16)
ax4.set_xlim(*ri_lim)
ax4.set_ylim(*iz_lim)

ax5.set_xlabel('i-y', fontsize = 27)
ax5.set_ylabel('i-z', fontsize = 27)
ax5.xaxis.set_tick_params(labelsize = 16)
ax5.yaxis.set_tick_params(labelsize = 16)
ax5.set_xlim(*iy_lim)
ax5.set_ylim(*iz_lim)
ax5.set_xlim(*iy_lim)
ax5.set_ylim(*iz_lim)


cbar = plt.colorbar(sdummy, ax=[ax3, ax4, ax5], orientation='horizontal',  extend ='both', shrink=0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('RF prob', fontsize = 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/noise_colorcolor.png', dpi=300, bbox_inches='tight')