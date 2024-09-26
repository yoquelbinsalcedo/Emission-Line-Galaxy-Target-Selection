import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy import table

crossmatch_output_name = 'desi2_pilot_hsc_wide_crossmatch'
crossmatch_output = f'{crossmatch_output_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_output}'))
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

# plot
fig, ax = plt.subplots()
ax.stairs(weightedavg, edges, linewidth=2, color ='grey', label = 'DESI LOP')
ax.stairs(normhist, edges, linewidth=2, color='blue', label="DESI-2 ELGs")
ax.axvline(x=1.1,ls='--', color='black')
ax.axvline(x=1.60,ls='--', color='black')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Spec z', fontsize=16)
ax.set_ylabel('Observed N [$deg^{-2}$]', fontsize=16)
ax.legend(loc='upper left', fontsize=16)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_pilot/desi1_desi2_elgs.png', dpi=300, bbox_inches='tight')
