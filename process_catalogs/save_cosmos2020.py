import numpy as np
from astropy import table
from pathlib import Path
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5


dir_in_cosmos = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/input/cosmos2020_farmer.fits'
cat = table.Table.read(dir_in_cosmos, format='fits', hdu=1)
colormaskx = np.logical_and.reduce((np.isfinite(cat['photoz']),
                                    np.isfinite(cat['CFHT_u_MAG']),
                                    np.isfinite(cat['HSC_g_MAG']),
                                    np.isfinite(cat['HSC_r_MAG']),
                                    np.isfinite(cat['HSC_i_MAG']),
                                    np.isfinite(cat['HSC_z_MAG']),
                                    np.isfinite(cat['HSC_y_MAG']),
                                    (np.logical_or(cat['HSC_g_MAG'] < 24.5, cat['HSC_r_MAG'] < 24.5))))

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
r_cos = cat['HSC_r_MAG']
g_cos = cat['HSC_g_MAG']
i_cos = cat['HSC_i_MAG']
y_cos = cat['HSC_y_MAG']
z_cos = cat['HSC_z_MAG']

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
cosmos_cat = cat[colormaskx]

# write to hdf5 file
output_name = 'cosmos2020'
output = f'{output_name}.hdf5'
hdf5_file_path = str(Path(f'../data/processed/hdf5/{output}'))
write_table_hdf5(cosmos_cat, output=hdf5_file_path, serialize_meta=True, overwrite=True)

# code to test if you can read in the catalog from hdf5 file:
test_hsc_table = read_table_hdf5(input=hdf5_file_path)
print(test_hsc_table['HSC_g_MAG'])
