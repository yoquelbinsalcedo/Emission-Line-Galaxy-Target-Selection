#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
import pandas as pd
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.io.misc.hdf5 import write_table_hdf5


# load in specz catalog we targeted 

dir_in_yoki = Path('../data/input/elgs_desi2_yoki.fits')
tert = table.Table.read(dir_in_yoki, format='fits', hdu=1)
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0
tert['EFFTIME'] = tert['TSNR2_LRG']*12.15
exposure = tert['EFFTIME']
tmask = exposure > 200
ysh = tert['YSH'] == True
final_masks = np.logical_and.reduce((elgmask, fiber_status, tmask, ysh))
desi2_pilot_cat = tert[final_masks]

# write to hdf5 file in hsc processed directory

output_name = 'desi2_pilot'
output = f'{output_name}.hdf5'
hdf5_file_path = str(Path(f'../data/processed/hdf5/{output}'))
write_table_hdf5(desi2_pilot_cat, output=hdf5_file_path, serialize_meta=True, overwrite=True)

#path=(hdf5_file_path)
# test if you can read in the catalog from hdf5 file:

test_hsc_table = read_table_hdf5(input=hdf5_file_path)
print(test_hsc_table['Z'])