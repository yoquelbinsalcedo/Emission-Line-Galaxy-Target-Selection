import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from astropy.io.misc.hdf5 import read_table_hdf5
from sklearn.model_selection import train_test_split
import pickle

z_range = np.array([1.3, 1.5])
path_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
path_model = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/hscwide_{z_range[0]}_{z_range[1]}_clf.pkl'
cat = read_table_hdf5(input=path_cat)

# quality cuts for reliable spec-z
o2_snr = cat['OII_FLUX'] * np.sqrt(cat['OII_FLUX_IVAR'])  
chi2 = cat['DELTACHI2']
mask_lop_good_z = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
mask_reliable_z = np.logical_or(mask_lop_good_z, chi2 > 25)
cat = cat[mask_reliable_z].to_pandas()


test_size = 0.20
y = np.logical_and(cat['Z'] > z_range[0], cat['Z'] < z_range[1])
X = cat[['g-r', 'g-i', 'g-z', 'g-y', 'r-i', 'r-z', 'r-y', 'i-z', 'i-y', 'z-y', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
specz = cat['Z']

X_train, X_test, specz_train, specz_test, y_train, y_test = train_test_split(X, specz, y, test_size=test_size)

# training the RF
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# performance
y_pred = clf.predict(X_test)
auc_score = roc_auc_score(y_test, y_pred)
print(f'The auc score for the {z_range[0]} < z < {z_range[1]} RF on all features is', auc_score)

# save the RF to a pickle file 
pickle.dump(clf, open(path_model, 'wb'))
