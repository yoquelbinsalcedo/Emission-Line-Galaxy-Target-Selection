import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy.io.misc.hdf5 import read_table_hdf5
import optimizecuts as oc
from importlib import reload
import pickle
reload(oc)

z_range = np.array([1.3, 1.5])
path_model = f'/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/hscwide_{z_range[0]}_{z_range[1]}_clf.pkl'
path_crossmatch_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
path_hsc_cat = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/hsc_wide_mags_failure_flags.hdf5'

cat_full = read_table_hdf5(input=path_hsc_cat).to_pandas()
cat_comb = read_table_hdf5(input=path_crossmatch_cat).to_pandas()

clf = pickle.load(open(path_model, 'rb'))

X_full = cat_full[['g-r', 'g-i', 'g-z', 'g-y', 'r-i', 'r-z', 'r-y', 'i-z', 'i-y', 'z-y', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
predict_prob_full = clf.predict_proba(X_full)[:, 1]

X_comb = cat_comb[['g-r', 'g-i', 'g-z', 'g-y', 'r-i', 'r-z', 'r-y', 'i-z', 'i-y', 'z-y', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'y_mag']]
predict_prob_comb = clf.predict_proba(X_comb)[:, 1]

# best case for 1.1 < z <  1.6 for rioffset, iyoffset, izmin, and gfiblim values
optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path)
gfiblim = opt_params['gfiberlim'][0]


# define the arrays to fill with our results
n_rf_prob = 20
rf_prob = np.linspace(0.025, 0.50, n_rf_prob)
zsuccess = np.zeros(n_rf_prob)
rangesuccess = np.zeros(n_rf_prob)
surface_density = np.zeros(n_rf_prob)
surf_density_act = np.zeros(n_rf_prob)

for i, rf_cut in enumerate(rf_prob):
    success_rates = oc.get_success_rate_rf(cat_comb, predict_prob_comb, rf_prob_cut=rf_cut, zrange=(1.3, 1.5), gfiblim=gfiblim)
    surf_density = oc.get_surf_density_rf(cat_full, predict_prob_full, rf_prob_cut=rf_cut, gfiblim=gfiblim)
    zsuccess[i] = success_rates[0]
    rangesuccess[i] = success_rates[1]
    surface_density[i] = surf_density
    surf_density_act[i] = surface_density[i] * rangesuccess[i]

    print(f'For RF prob => {rf_cut}: \n target density = {surf_density} \n z success rate = {zsuccess[i]} \n z range success rate = {rangesuccess[i]} \n net surface density yield = {surf_density_act[i]}')


# here, we are going to plot a panel with the target density, net surface density yield, and redshift range success rate


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(29, 10), tight_layout=True)
plt.subplots_adjust(wspace=0.82)

ax0.plot(rf_prob, surface_density, color = 'black', lw=4, zorder=1)
ax0.set_xlabel('RF prob cut (=>)', fontsize = 32)
ax0.set_ylabel(r'Target density ($\Sigma_\mathrm{target}$)', fontsize = 31)
ax0.xaxis.set_tick_params(labelsize = 29)
ax0.yaxis.set_tick_params(labelsize = 29)
#ax0.legend(loc = 'upper left', fontsize = 30)

ax1.plot(rf_prob, rangesuccess, color = 'black', lw=4, zorder=1)
ax1.set_xlabel('RF prob cut (=>)', fontsize = 32)
ax1.set_ylabel(r'Redshift range success rate ($f_\mathrm{z\,= \, 1.3\, -\, 1.5}^{700 \,<\,t\, <\,1400}$)', fontsize = 31)
ax1.set_ylim(0,1)
ax1.xaxis.set_tick_params(labelsize = 29)
ax1.yaxis.set_tick_params(labelsize = 29)
# ax1.axvline(x=gfiblim,ls= '--',color='purple', lw=4, zorder=2)
# ax1.plot(24.21, 0.34, marker = '*', color = 'grey', markersize = 30, linestyle='None', zorder=3)
# ax1.plot(gfiblim, 0.84, marker = '*', color = 'blue', markersize = 30, linestyle='None', zorder=4)


ax2.plot(rf_prob, surf_density_act, color = 'black', lw=4, zorder=1)
ax2.set_xlabel('RF prob cut (=>)', fontsize = 32)
ax2.set_ylabel(r'Net surface density yield ($\Sigma_\mathrm{yield}^{z\,= \, 1.3\, -\, 1.5}$)', fontsize = 31)
ax2.xaxis.set_tick_params(labelsize = 29)
ax2.yaxis.set_tick_params(labelsize = 29)
# ax2.axvline(x=gfiblim,ls= '--',color='purple', lw=4, zorder=2)
# ax2.plot(24.21, 660, marker = '*', color = 'grey', markersize = 30, linestyle='None')
# ax2.plot(gfiblim, 1372, marker = '*', color = 'blue', markersize = 30, linestyle='None', zorder=3)

plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/analysis/figures/truth/rf_1.1_z_1.3_sum_stats_3panel.png', dpi=300, bbox_inches='tight')