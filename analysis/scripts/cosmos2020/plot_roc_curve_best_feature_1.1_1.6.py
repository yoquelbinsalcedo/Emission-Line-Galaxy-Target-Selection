import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import roc_curve
from astropy.io.misc.hdf5 import read_table_hdf5

z_range = np.array([1.1, 1.6])
path_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/cosmos2020.hdf5'
path_best_features = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/best_rf_features/best_{z_range[0]}_{z_range[1]}.npy'
path_best_model = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/cosmos2020_{z_range[0]}_{z_range[1]}_clf_best_features.pkl'
path_fig = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/analysis/figures/cosmos2020/roc_curve_best_feature_{z_range[0]}_{z_range[1]}.png'


cat_name = 'cosmos2020'
file_name = f'{cat_name}.hdf5'
cat = read_table_hdf5(input=path_cat).to_pandas()
best_features = np.load(path_best_features)


test_size = 0.20
y = np.logical_and(cat['photoz'] > z_range[0], cat['photoz'] < z_range[1])
X = cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy', 'HSC_r_MAG', 'HSC_i_MAG', 'HSC_y_MAG', 'HSC_z_MAG', 'CFHT_u_MAG', 'HSC_g_MAG']]
photz = cat['photoz']
clf = pickle.load(open(path_best_model, 'rb'))

X_train, X_test, photz_train, photz_test, y_train, y_test = train_test_split(X, photz, y, test_size=0.20)
predict_prob_test = clf.predict_proba(X_test[[f'{best_features[0]}', f'{best_features[1]}', f'{best_features[2]}']])[:, 1]
rf_prob_cutoff = 0.025


# look at the roc curve for the test set of 1.1 < z < 1.6 cosmos2020 galaxies
fpr, tpr, thresholds = roc_curve(y_test, predict_prob_test)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'{z_range[0]} < z < {z_range[1]} with {best_features} on test set')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.axvline(x=rf_prob_cutoff, ls='--', color='black', label=f'FPR = {rf_prob_cutoff}')
ax.legend(fontsize=18)
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
