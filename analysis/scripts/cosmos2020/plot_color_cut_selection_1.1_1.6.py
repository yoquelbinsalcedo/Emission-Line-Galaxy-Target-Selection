import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from astropy.io.misc.hdf5 import read_table_hdf5
import colorcet as cc

def cmap_white(cmap_name):
    """Returns a colormap with white as the lowest value color."""
    import numpy as np
    try:
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        cmap = cm.get_cmap(cmap_name, 256)
    except ValueError:
        import seaborn as sns
        cmap = sns.color_palette("flare", as_cmap=True)
    newcolors = cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 0])
    newcolors[:1, :] = white
    cmap_white = ListedColormap(newcolors)
    return cmap_white

z_range = np.array([1.1, 1.6])
path_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/cosmos2020.hdf5'
path_best_features = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/best_rf_features/best_{z_range[0]}_{z_range[1]}.npy'
path_fig = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/analysis/figures/cosmos2020/color_cut_selection_{z_range[0]}_{z_range[1]}.png'

cat_name = 'cosmos2020'
file_name = f'{cat_name}.hdf5'
cat = read_table_hdf5(input=path_cat).to_pandas()
best_features = np.load(path_best_features)


X = cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy', 'HSC_r_MAG', 'HSC_i_MAG', 'HSC_y_MAG', 'HSC_z_MAG', 'CFHT_u_MAG', 'HSC_g_MAG']]
photz = cat['photoz']

# what to plot
iy, ri, iz = X[f'{best_features[0]}'], X[f'{best_features[1]}'], X[f'{best_features[2]}']
iy_cut = 0.35
iy_ri_cut = - 0.19
color_mask = np.logical_and(iy + iy_ri_cut > ri, iy > iy_cut)


# colormap to use
cmap = cc.cm.kbc
cmap.set_extremes(under='red', over='springgreen')
vmin = 1.1
vmax = 1.6

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(24, 10))


sd0 = ax0.scatter(ri[color_mask], iy[color_mask], c=photz[color_mask], cmap=cmap, s=12, alpha=0.1, vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri[color_mask], iz[color_mask], c=photz[color_mask], cmap=cmap, s=12, alpha=0.1, vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy[color_mask], iz[color_mask], c=photz[color_mask], cmap=cmap, s=12, alpha=0.1, vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x=100, y=100, c=photz[color_mask].iloc[0], cmap=cmap, s=12, alpha=1, vmin=vmin, vmax=vmax)

ri_lim = (-0.5, 1.5)
iy_lim = (0, 1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize=27)
ax0.set_ylabel('i-y', fontsize=27)
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
ax0.xaxis.set_tick_params(labelsize=16)
ax0.yaxis.set_tick_params(labelsize=16)

ax1.set_xlabel('r-i', fontsize=27)
ax1.set_ylabel('i-z', fontsize=27)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)

ax2.set_xlabel('i-y', fontsize=27)
ax2.set_ylabel('i-z', fontsize=27)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)


cbar = plt.colorbar(sdummy, ax=[ax0, ax1, ax2], orientation='horizontal', extend='both', shrink=0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label(f'LePhare photo-z', fontsize=27, loc='center')
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
