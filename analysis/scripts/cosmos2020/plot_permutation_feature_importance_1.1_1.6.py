import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pickle
from astropy.io.misc.hdf5 import read_table_hdf5

z_range = np.array([1.1, 1.6])
path_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/cosmos2020.hdf5'
path_model = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/cosmos2020_{z_range[0]}_{z_range[1]}_clf_all_features.pkl'
path_fig = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/analysis/figures/cosmos2020/feature_permutation_importance.png'
path_best_features = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/best_rf_features/best_{z_range[0]}_{z_range[1]}.npy'


cat_name = 'cosmos2020'
file_name = f'{cat_name}.hdf5'
cat = read_table_hdf5(input=path_cat).to_pandas()


y = np.logical_and(cat['photoz'] > z_range[0], cat['photoz'] < z_range[1])
X = cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy', 'HSC_r_MAG', 'HSC_i_MAG', 'HSC_y_MAG', 'HSC_z_MAG', 'CFHT_u_MAG', 'HSC_g_MAG']]
photz = cat['photoz']
clf = pickle.load(open(path_model, 'rb'))

# use feature permuation to determine which features are most important and plot them for the redshift range of interest
result = permutation_importance(clf, X, y)
sorted_idx = result.importances_mean.argsort()

# visualize the feature importance and label each feature for the given redshift cut
fig, ax = plt.subplots(figsize=(12,8))
ax.boxplot(result.importances[sorted_idx].T, vert=False, tick_labels=X.columns[sorted_idx])
ax.set_title(f"Permutation Importance COSMOS2020 {z_range[0]} < z < {z_range[1]}", fontsize=18)
fig.tight_layout()
plt.savefig(path_fig, dpi=300)

# save 3 best features to file to call in later and train new RF model on
best_features = ['iy', 'ri', 'iz']
np.save(path_best_features, best_features)
