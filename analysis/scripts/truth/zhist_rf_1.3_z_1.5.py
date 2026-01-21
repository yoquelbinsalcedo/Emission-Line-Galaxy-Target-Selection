import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import table
from astropy.io.misc.hdf5 import read_table_hdf5
import pickle


z_range = np.array([1.3, 1.5])
path_model = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/hscwide_{z_range[0]}_{z_range[1]}_clf.pkl'
path_crossmatch = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=path_crossmatch).to_pandas()

# quality cuts for reliable spec-z
o2_snr = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR'])  
chi2 = combined_cat['DELTACHI2']
mask_lop_good_z = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
mask_reliable_z = np.logical_or(mask_lop_good_z, chi2 > 25)
combined_cat = combined_cat[mask_reliable_z]
specz = combined_cat['Z']

clf = pickle.load(open(path_model, 'rb'))
X_comb = combined_cat[['g-r', 'g-i', 'g-z', 'g-y', 'r-i', 'r-z', 'r-y', 'i-z', 'i-y', 'z-y', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
predict_prob_comb = clf.predict_proba(X_comb)[:, 1]
mask_rf_prob = predict_prob_comb >= 0.39999999999999997

# best case for 1.1 < z <  1.6 gfiblim value
optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path)
gfiblim = opt_params['gfiberlim'][0]

# load in the desi ELG distributions to plot against our final distribution
data = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/desi_elg_ts_zenodo/main-800coaddefftime1200-nz-zenodo.ecsv', format='ascii.ecsv')
data.colnames
zmin = data['ZMIN']
zmax = data['ZMAX']
lop_north = data['ELG_LOP_NORTH']
lop_south_decal = data['ELG_LOP_SOUTH_DECALS']
lop_south_des = data['ELG_LOP_SOUTH_DES']
vlo_north = data['ELG_VLO_NORTH']
vlo_south_decal = data['ELG_VLO_SOUTH_DECALS']
vlo_south_des = data['ELG_VLO_SOUTH_DES']
lop_desi = data['ELG_LOP_DESI']
vlo_desi = data['ELG_VLO_DESI']
# - {AREA_NORTH: 4400}
# - {AREA_SOUTH_DECALS: 8500}
# - {AREA_SOUTH_DES: 1100}
weightedavg = (lop_north * 4400 + lop_south_decal * 8500 + lop_south_des * 1100 )/(14000)
fiber_assign_rate = 0.69
weightedavg = weightedavg / fiber_assign_rate

# normalize the elgs to sum up to the total number of galaxies with good redshifts per square degrees

values, edges = np.histogram(specz[mask_rf_prob], bins=np.linspace(0, 2, 41))
wrongnorm = np.sum(values)
rightnorm = (1926 * 0.80) # product of z success rate and target surface density
normhist = values * (rightnorm/wrongnorm)


fig, ax = plt.subplots()
# ax.grid(False)
ax.stairs(weightedavg, edges, linewidth=4, color ='grey', label= 'DESI ELGs')
ax.stairs(normhist, edges, linewidth=4, color='Blue', label='RF selection')
ax.axvline(x=1.3,ls='--', color='black')
ax.axvline(x=1.5,ls='--', color='black')
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_xlabel('$z_{\mathrm{spec}}$', fontsize=35)
ax.set_ylabel('Targets [$\mathrm{deg^{-2}}$]', fontsize=35)
ax.legend(loc='upper left', fontsize=26)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/analysis/figures/truth/zhist_rf_1.3_z_1.5.png', dpi=300, bbox_inches='tight')