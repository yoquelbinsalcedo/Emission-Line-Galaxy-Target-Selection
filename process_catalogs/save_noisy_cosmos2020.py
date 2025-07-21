import numpy as np
from astropy import table
from pathlib import Path
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5

# load in cosmos2020 catalog
catversion = 'farmer'
cosmos_dir_in = Path(f'../data/input/cosmos2020_{catversion}.fits')
cat = table.Table.read(cosmos_dir_in, format='fits', hdu=1)

# We create an array containing a noisy version of the y magnitudes (m_noisy) by taking the array of the non-noisy y magntiudes (m_initial) 
# and adding the array of sigmas multiplied by an array of equal length consisting of N random numbers distributed as a Gaussian with mean 0 and sigma 1
# We’d want to use LSST year 2 depths to calculate predict_probably noise LSST y band depth +2.5*log10(year/10) with year = 2
# LSST noise year y band depth are y : 22.96, 24.8 for single image and coadded images


r_cos = cat['HSC_r_MAG']
g_cos = cat['HSC_g_MAG']
i_cos = cat['HSC_i_MAG']
y_cos = cat['HSC_y_MAG']
z_cos = cat['HSC_z_MAG']

ug_cos = cat['CFHT_u_MAG'] - cat['HSC_g_MAG'] 
ur_cos = cat['CFHT_u_MAG'] - cat['HSC_r_MAG']
ui_cos = cat['CFHT_u_MAG'] - cat['HSC_i_MAG']
uz_cos = cat['CFHT_u_MAG'] - cat['HSC_z_MAG'] 
uy_cos = cat['CFHT_u_MAG'] - cat['HSC_y_MAG']
gr_cos = cat['HSC_g_MAG'] - cat['HSC_r_MAG']
gi_cos = cat['HSC_g_MAG'] - cat['HSC_i_MAG']
gz_cos = cat['HSC_g_MAG'] - cat['HSC_z_MAG']
gy_cos = cat['HSC_g_MAG'] - cat['HSC_y_MAG']
ri_cos = cat['HSC_r_MAG'] - cat['HSC_i_MAG']
rz_cos = cat['HSC_r_MAG'] - cat['HSC_z_MAG']
ry_cos = cat['HSC_r_MAG'] - cat['HSC_y_MAG']
iz_cos = cat['HSC_i_MAG'] - cat['HSC_z_MAG']
iy_cos = cat['HSC_i_MAG'] - cat['HSC_y_MAG']
zy_cos = cat['HSC_z_MAG'] - cat['HSC_y_MAG']

cat['ug'] = ug_cos
cat['ur'] = ur_cos
cat['ui'] = ui_cos
cat['uz'] = uz_cos
cat['uy'] = uy_cos
cat['gr'] = gr_cos
cat['gi'] = gi_cos
cat['gz'] = gz_cos
cat['gy'] = gy_cos
cat['ri'] = ri_cos
cat['rz'] = rz_cos
cat['ry'] = ry_cos
cat['iz'] = iz_cos
cat['iy'] = iy_cos
cat['zy'] = zy_cos
cat['HSC_r_MAG'] = r_cos
cat['HSC_g_MAG'] = g_cos
cat['HSC_i_MAG'] = i_cos
cat['HSC_y_MAG'] = y_cos
cat['HSC_z_MAG'] = z_cos

# function to calculate sigma for each magnitude


def sigma_mag(m_5, xmag):
    '''m_5: LSST year 2 band depth
       x_mag: array of magnitudes of the band you want to calculate sigma for
    '''
    flux_5_f = 10**(((xmag - m_5) / 2.5))
    sigma_f_f = 1/5*flux_5_f
    sigma_m = (2.5/np.log(10))*sigma_f_f
    return sigma_m


bands = ['g', 'r', 'i', 'z', 'y']
five_sig_depth = [26.9, 26.9, 26.4, 25.6, 24.8]  # these are coadded

for band, depth in zip(bands, five_sig_depth):

    year2_depth = (depth + 1.25*np.log10(2/10))  # LSST year 2 depth y band
    sigma = sigma_mag(year2_depth, cat[f'HSC_{band}_MAG'])
    gauss = np.random.normal(0, 1, len(cat[f'HSC_{band}_MAG']))
    cat[f'{band}_mag_noisy'] = cat[f'HSC_{band}_MAG'] + sigma * gauss



# define colors based on noisy photometry (except u band) (u-g-r-i-z-y)
g_noise = cat['g_mag_noisy']
r_noise = cat['r_mag_noisy']
i_noise = cat['i_mag_noisy']
z_noise = cat['z_mag_noisy']
y_noise = cat['y_mag_noisy']

ug_noise = cat['CFHT_u_MAG'] - cat['g_mag_noisy'] 
ur_noise = cat['CFHT_u_MAG'] - cat['r_mag_noisy']
ui_noise = cat['CFHT_u_MAG'] - cat['i_mag_noisy']
uz_noise = cat['CFHT_u_MAG'] - cat['z_mag_noisy'] 
uy_noise = cat['CFHT_u_MAG'] - cat['y_mag_noisy']
gr_noise = cat['g_mag_noisy'] - cat['r_mag_noisy']
gi_noise = cat['g_mag_noisy'] - cat['i_mag_noisy']
gz_noise = cat['g_mag_noisy'] - cat['z_mag_noisy']
gy_noise = cat['g_mag_noisy'] - cat['y_mag_noisy']
ri_noise = cat['r_mag_noisy'] - cat['i_mag_noisy']
rz_noise = cat['r_mag_noisy'] - cat['z_mag_noisy']
ry_noise = cat['r_mag_noisy'] - cat['y_mag_noisy']
iz_noise = cat['i_mag_noisy'] - cat['z_mag_noisy']
iy_noise = cat['i_mag_noisy'] - cat['y_mag_noisy']
zy_noise = cat['z_mag_noisy'] - cat['y_mag_noisy']

cat['g_mag_noisy'] = g_noise
cat['r_mag_noisy'] = r_noise
cat['i_mag_noisy'] = i_noise
cat['z_mag_noisy'] = z_noise
cat['y_mag_noisy'] = y_noise

cat['ug_noisy'] = ug_noise 
cat['ur_noisy'] = ur_noise 
cat['ui_noisy'] = ui_noise 
cat['uz_noisy'] = uz_noise 
cat['uy_noisy'] = uy_noise 
cat['gr_noisy'] = gr_noise 
cat['gi_noisy'] = gi_noise 
cat['gz_noisy'] = gz_noise 
cat['gy_noisy'] = gy_noise 
cat['ri_noisy'] = ri_noise 
cat['rz_noisy'] = rz_noise 
cat['ry_noisy'] = ry_noise 
cat['iz_noisy'] = iz_noise 
cat['iy_noisy'] = iy_noise 
cat['zy_noisy'] = zy_noise


colormask = np.logical_and.reduce((np.isfinite(cat['HSC_r_MAG']),
                                    np.isfinite(cat['photoz']),
                                    np.isfinite(cat['HSC_g_MAG']),
                                    np.isfinite(cat['HSC_i_MAG']),
                                    np.isfinite(cat['HSC_y_MAG']),
                                    np.isfinite(cat['CFHT_u_MAG']),
                                    np.isfinite(cat['HSC_z_MAG']),
                                    (np.logical_or(cat['HSC_g_MAG'] < 24.5, cat['HSC_r_MAG'] < 24.5))))

cosmos_cat = cat[colormask]


# write to hdf5 file
output_name = 'noisy_cosmos2020'
output = f'{output_name}.hdf5'
output_file_path = str(Path(f'../data/processed/hdf5/{output}'))
write_table_hdf5(cosmos_cat, output=output_file_path, serialize_meta=True, overwrite=True)

# code to test if you can read in the catalog from hdf5 file:
test_hsc_table = read_table_hdf5(input=output_file_path)
print(test_hsc_table['z_mag_noisy'])
