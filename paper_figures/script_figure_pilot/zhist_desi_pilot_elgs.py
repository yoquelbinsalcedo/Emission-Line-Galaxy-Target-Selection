import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy import table


crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/desi2_pilot_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)

# masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR']) 
chi2_comb = combined_cat['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail = ~ snr_mask_comb

# best cuts for 1.1 < z < 1.6 case
rishift, iyshift, izmin, gfiblim = 0, 5.979861E-02, 3.634577E-01, 2.422445E+01

spectro_z = combined_cat['Z'][snr_mask_comb]
target_iyri_cut = -0.19
target_iy_cut = 0.35
color_mask_iyri = np.logical_and(
    (combined_cat['r_mag'][snr_mask_comb] - combined_cat['i_mag'][snr_mask_comb] < combined_cat['i_mag'][snr_mask_comb] - combined_cat['y_mag'][snr_mask_comb] + target_iyri_cut + rishift),
    (combined_cat['i_mag'][snr_mask_comb] - combined_cat['y_mag'][snr_mask_comb] > target_iy_cut + iyshift))
color_mask_iz = (combined_cat['i_mag'][snr_mask_comb] - combined_cat['z_mag'][snr_mask_comb]) > izmin
gfiber_mask = combined_cat['g_fiber_mag'][snr_mask_comb] < gfiblim
ccuts = np.logical_and.reduce((color_mask_iyri, color_mask_iz, gfiber_mask))

# spec truth sample
crossmatch_file_path_truth = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat_truth = read_table_hdf5(input=crossmatch_file_path_truth)

# quality cuts for reliable spec-z
o2_snr_rongpu = combined_cat_truth['OII_FLUX'] * np.sqrt(combined_cat_truth['OII_FLUX_IVAR'])  
chi2_rongpu = combined_cat_truth['DELTACHI2']
o2_chi2_mask_rongpu = o2_snr_rongpu > 10**(0.9 - 0.2*np.log10(chi2_rongpu))
snr_mask_rongpu = np.logical_or(o2_chi2_mask_rongpu, chi2_rongpu > 25)
snr_mask_rongpu_fail = ~ snr_mask_rongpu


# photometry for reliable spec-z
ri_rongpu = combined_cat_truth['r_mag'][snr_mask_rongpu] - combined_cat_truth['i_mag'][snr_mask_rongpu]
iz_rongpu = combined_cat_truth['i_mag'][snr_mask_rongpu] - combined_cat_truth['z_mag'][snr_mask_rongpu]
iy_rongpu = combined_cat_truth['i_mag'][snr_mask_rongpu] - combined_cat_truth['y_mag'][snr_mask_rongpu]
gfiber_rongpu = combined_cat_truth['g_fiber_mag'][snr_mask_rongpu]

specz_rongpu = combined_cat_truth['Z'][snr_mask_rongpu]

# best case for 1.1 < z <  1.6 for rioffset, iyoffset, izmin, and gfiblim values

optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'
opt_params = pd.read_csv(optimization_csv_path)
rishift_rongpu, iyshift_rongpu, izmin_rongpu, gfiblim_rongpu = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]

target_iyri_cut = -0.19
target_iy_cut = 0.35
color_mask_iyri_rongpu = np.logical_and(
    (ri_rongpu < iy_rongpu + target_iyri_cut + rishift_rongpu),
    (iy_rongpu > target_iy_cut + iyshift_rongpu))
color_mask_iz_rongpu = (iz_rongpu) > izmin_rongpu
gfiber_mask_rongpu = gfiber_rongpu < gfiblim_rongpu
ccuts_rongpu = np.logical_and.reduce((color_mask_iyri_rongpu, color_mask_iz_rongpu, gfiber_mask_rongpu))

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

# normalize
values, edges = np.histogram(spectro_z[ccuts], bins=np.linspace(0, 2, 41))
wrongnorm = np.sum(values)
rightnorm = 0.89 * 1762  # z success rate * target density
normhist = values * (rightnorm/wrongnorm)

values_rongpu, edges_rongpu = np.histogram(specz_rongpu[ccuts_rongpu], bins=np.linspace(0, 2, 41))
wrongnorm_rongpu = np.sum(values_rongpu)
rightnorm_rongpu = (1633 * 0.89) # product of z success rate and target surface density
normhist_rongpu = values_rongpu * (rightnorm_rongpu/wrongnorm_rongpu)

# plot
fig, ax = plt.subplots()
ax.stairs(weightedavg, edges, linewidth=4, color ='grey', label = 'DESI LOP')
ax.stairs(normhist_rongpu, edges_rongpu, linewidth=4, color='blue', label='Spectroscopic truth')
ax.stairs(normhist, edges, linewidth=4, color='orange', label="DESI-2 pilot")
ax.axvline(x=1.1,ls='--', color='black')
ax.axvline(x=1.60,ls='--', color='black')
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_xlabel('$z_{\mathrm{spec}}$', fontsize=35)
ax.set_ylabel('Observed N [$\mathrm{deg^{-2}}$]', fontsize=35)
ax.legend(loc='upper left', fontsize=26)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paper_figures/script_figure_pilot/zhist_desi_pilot_elgs.png', dpi=300, bbox_inches='tight')
