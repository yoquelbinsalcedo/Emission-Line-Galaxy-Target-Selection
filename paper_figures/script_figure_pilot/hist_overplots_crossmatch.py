import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/hsc_wide_cosmos2020_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path).to_pandas()


yb_comb = np.logical_and(combined_cat['photoz'] > 1.1, combined_cat['photoz'] < 1.6)
zallb_comb = combined_cat['photoz']
xallb_comb = combined_cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
gmag_comb = combined_cat['g_mag']
rmag_comb = combined_cat['r_mag']
imag_comb = combined_cat['i_mag']
ymag_comb = combined_cat['y_mag']
zmag_comb = combined_cat['z_mag']
gfib_comb = combined_cat['g_fiber_mag']
rfib_comb = combined_cat['r_fiber_mag']
xallb_comb_train, xallb_comb_test, gfib_comb_train, gfib_comb_test, rfib_comb_train, rfib_comb_test, gmag_comb_train, gmag_comb_test, rmag_comb_train, rmag_comb_test, imag_comb_train, imag_comb_test, zmag_comb_train, zmag_comb_test, ymag_comb_train, yallb_comb_test, yb_comb_train, yb_comb_test, zallb_comb_train, zallb_comb_test = train_test_split(xallb_comb, gfib_comb, rfib_comb, gmag_comb, rmag_comb, imag_comb, zmag_comb, ymag_comb, yb_comb, zmag_comb)

# train the RF over 1.1 < z < 1.6 
clfallb_comb = RandomForestClassifier()
clfallb_comb.fit(xallb_comb_train, yb_comb_train)

# # save the RF to a pickle file
# model_save_path = Path('../data/models/clfallb_comb.pkl')
# pickle.dump(clfallb_comb, open(model_save_path, 'wb'))


pickled_clfallb_comb = pickle.load(open('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/clfallb_comb.pkl', 'rb'))
proballb_comb = pickled_clfallb_comb.predict_proba(xallb_comb)
probmaskallb_comb = proballb_comb[:, 1] >= 0.025

# define colors to plot
iz = imag_comb - zmag_comb
ri = rmag_comb - imag_comb
iy = imag_comb - ymag_comb


# color cuts
cuts = np.logical_and(iy - 0.19  > ri, iy > 0.35)
not_color =~ cuts
cuts_iz = np.logical_and.reduce((iy - 0.19 > ri, iy > 0.35, iz > 0.374))
cuts_iz_only = iz > 0.374
cuts_rf = proballb_comb[:, 1] >= 0.025
cuts_rf50 = proballb_comb[:, 1] >= 0.50
not_rf =~ cuts_rf

# objects that did not meet color cuts but have RF_prob > 0.025
not_color_rf = np.logical_and(not_color, cuts_rf )

# final sample for spectra cuts
gfib_mask = gfib_comb < 24.3
gband = np.logical_and(gmag_comb < 24, gfib_mask)
rfib_mask = gfib_comb < 24.3
rband = np.logical_and(rmag_comb < 24, rfib_mask)
inlimit = np.logical_or(rband, gband)
cuts_iyri = np.logical_and((iy > 0.35),(ri < iy - 0.19)) 
cuts_final = np.logical_and(cuts_iyri, inlimit)

# photoz
photoz = zallb_comb

# plot the redshift distribution for the different cuts and save
fig, ax = plt.subplots()
ax.hist(photoz[cuts], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'lightgrey', fill = True, label = 'i-y/r-i cut')
ax.hist(photoz[cuts_iz], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'lightblue', fill = True ,label = '+ i-z cut')
ax.hist(photoz[cuts_rf], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'red', label = 'RF prob > 0.025')
ax.axvline(x = 1.1,ls= '--', color='black')
ax.axvline(x = 1.6,ls= '--', color='black')
ax.set_xlabel('LePhare Photometric Redshift', fontsize = 20)
ax.set_ylabel('Count', fontsize = 20)
ax.xaxis.set_tick_params(labelsize = 16)
ax.yaxis.set_tick_params(labelsize = 16)
ax.legend(loc = 'upper left', fontsize = 18)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_pilot/hist_overplots_crossmatch.png', dpi=300, bbox_inches='tight')