import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import table
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

# load in crossmatched cat of hsc wide and spec truth
crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat = read_table_hdf5(input=crossmatch_file_path)

# quality cuts for reliable spec-z
o2_snr_rongpu = combined_cat['OII_FLUX'] * np.sqrt(combined_cat['OII_FLUX_IVAR'])  
chi2_rongpu = combined_cat['DELTACHI2']
o2_chi2_mask_rongpu = o2_snr_rongpu > 10**(0.9 - 0.2*np.log10(chi2_rongpu))
snr_mask_rongpu = np.logical_or(o2_chi2_mask_rongpu, chi2_rongpu > 25)
snr_mask_rongpu_fail = ~ snr_mask_rongpu


# photometry for reliable spec-z
ri_rongpu = combined_cat['r_mag'][snr_mask_rongpu] - combined_cat['i_mag'][snr_mask_rongpu]
iz_rongpu = combined_cat['i_mag'][snr_mask_rongpu] - combined_cat['z_mag'][snr_mask_rongpu]
iy_rongpu = combined_cat['i_mag'][snr_mask_rongpu] - combined_cat['y_mag'][snr_mask_rongpu]
gfiber_rongpu = combined_cat['g_fiber_mag'][snr_mask_rongpu]

specz_rongpu = combined_cat['Z'][snr_mask_rongpu]

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

# normalize the elgs to sum up to the total number of galaxies with good redshifts per square degrees

values_rongpu, edges = np.histogram(specz_rongpu[ccuts_rongpu], bins=np.linspace(0, 2, 41))
wrongnorm_rongpu = np.sum(values_rongpu)
rightnorm_rongpu = (1633 * 0.89) # product of z success rate and target surface density
normhist_rongpu = values_rongpu * (rightnorm_rongpu/wrongnorm_rongpu)



fig, ax = plt.subplots()
# ax.grid(False)
ax.stairs(weightedavg, edges, linewidth=4, color ='grey', label= 'DESI ELGs')
ax.stairs(normhist_rongpu, edges, linewidth=4, color='Blue', label='This work')
ax.axvline(x=1.10,ls='--', color='black')
ax.axvline(x=1.60,ls='--', color='black')
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.set_xlabel('$z_{\mathrm{spec}}$', fontsize=35)
ax.set_ylabel('Observed N [$\mathrm{deg^{-2}}$]', fontsize=35)
ax.legend(loc='upper left', fontsize=26)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paper_figures/script_figure_truth/zhist_desi_truth_elgs.png', dpi=300, bbox_inches='tight')