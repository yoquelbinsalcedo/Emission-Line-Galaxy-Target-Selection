import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
from astropy.io.misc.hdf5 import read_table_hdf5

z_range = np.array([1.1, 1.6])
path_model = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/cosmos2020_{z_range[0]}_{z_range[1]}_clf_all_features.pkl'
path_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/cosmos2020.hdf5'


cat_name = 'cosmos2020'
file_name = f'{cat_name}.hdf5'
cat = read_table_hdf5(input=path_cat).to_pandas()


test_size = 0.20
y = np.logical_and(cat['photoz'] > z_range[0], cat['photoz'] < z_range[1])
X = cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy', 'HSC_r_MAG', 'HSC_i_MAG', 'HSC_y_MAG', 'HSC_z_MAG', 'CFHT_u_MAG', 'HSC_g_MAG']]
photz = cat['photoz']


X_train, X_test, photz_train, photz_test, y_train, y_test = train_test_split(X, photz, y, test_size=test_size)

# training the RF
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# performance
y_pred = clf.predict(X_test)
auc_score = roc_auc_score(y_test, y_pred)
print(f'The auc score for the {z_range[0]} < z < {z_range[1]} RF on all features is', auc_score)

# save the RF to a pickle file 
pickle.dump(clf, open(path_model, 'wb'))
