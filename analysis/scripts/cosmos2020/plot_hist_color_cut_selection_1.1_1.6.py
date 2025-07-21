import numpy as np
from matplotlib import pyplot as plt
import pickle
from astropy.io.misc.hdf5 import read_table_hdf5
import colorcet as cc


def get_complete_purity(cuts, photozs, z_range=[1.1, 1.6]):

    zrange = np.logical_and(photozs > z_range[0], photozs < z_range[1])
    completeness = np.sum(np.logical_and(cuts, zrange)) / np.sum(zrange)
    purity = np.sum(np.logical_and(cuts, zrange)) / np.sum(cuts)

    return completeness, purity


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
path_fig = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/analysis/figures/cosmos2020/hist_color_cut_selection_{z_range[0]}_{z_range[1]}.png'

cat_name = 'cosmos2020'
file_name = f'{cat_name}.hdf5'
cat = read_table_hdf5(input=path_cat).to_pandas()
best_features = np.load(path_best_features)


X = cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy', 'HSC_r_MAG', 'HSC_i_MAG', 'HSC_y_MAG', 'HSC_z_MAG', 'CFHT_u_MAG', 'HSC_g_MAG']]
photz = cat['photoz']

# what to plot
iy, ri, iz = X[f'{best_features[0]}'], X[f'{best_features[1]}'], X[f'{best_features[2]}']
iy_cut = 0.35
iy_ri_cut = -0.19
color_mask = np.logical_and(iy + iy_ri_cut > ri, iy > iy_cut)

comp, pure = get_complete_purity(color_mask, photozs=photz)

zmin = 1.1
zmax = 1.6

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title(f'Comp= {comp:.2}, Pure = {pure:.2}', fontsize = 15, pad=10)
ax.hist(photz[color_mask], bins=np.linspace(0, 2, 100), histtype='step', color='black', label=f'Best color cut selection')
ax.axvline(x=zmin, ls='--', color='black')
ax.axvline(x=zmax, ls='--', color='black')
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
ax.set_xlabel('LePhare photo-z', fontsize=22)
ax.set_ylabel('Count', fontsize=22)
ax.legend(fontsize=18, loc='best')
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
