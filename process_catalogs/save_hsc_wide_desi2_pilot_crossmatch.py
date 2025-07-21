import numpy as np
from astropy import table
import pandas as pd 
from pathlib import Path
import h5py
from astropy.coordinates import SkyCoord
from astropy.table import hstack
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5
import astropy.units as u


# load in catalogs from hdf5 files

pilot_input_name = 'desi2_pilot'
pilot_input = f'{pilot_input_name}.hdf5'
pilot_file_path = str(Path(f'../data/processed/hdf5/{pilot_input}'))
pilot_cat = read_table_hdf5(input=pilot_file_path)

hsc_wide_input_name = 'hsc_wide_mags_failure_flags'
hsc_wide_input = f'{hsc_wide_input_name}.hdf5'
hsc_wide_file_path = str(Path(f'../data/processed/hdf5/{hsc_wide_input}'))
hsc_wide_cat = read_table_hdf5(input=hsc_wide_file_path)

# cross-match with HSC-W for photometry

ra_rongpu = pilot_cat['TARGET_RA']
dec_rongpu = pilot_cat['TARGET_DEC']
ra_hsc = hsc_wide_cat['ra']
dec_hsc = hsc_wide_cat['dec']

hsc_coord = SkyCoord(ra_hsc*u.degree, dec_hsc*u.degree)
elg_coord_rongpu = SkyCoord(ra_rongpu*u.degree, dec_rongpu*u.degree)
idx_r, d2d_r, d3d_r = elg_coord_rongpu.match_to_catalog_sky(hsc_coord)
dmask_r = d2d_r.arcsec < 1
hsc_wide_pilot_cat = hstack([pilot_cat, hsc_wide_cat[idx_r]])[dmask_r]

# write to hdf5 file in hsc processed directory

crossmatch_output_name = 'desi2_pilot_hsc_wide_crossmatch'
crossmatch_output = f'{crossmatch_output_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_output}'))
write_table_hdf5(hsc_wide_pilot_cat, output=crossmatch_file_path, serialize_meta=True, overwrite=True)


# code to test if you can read in the catalog from hdf5 file:

test_hsc_table = read_table_hdf5(input=crossmatch_file_path)
print(test_hsc_table['Z'])
