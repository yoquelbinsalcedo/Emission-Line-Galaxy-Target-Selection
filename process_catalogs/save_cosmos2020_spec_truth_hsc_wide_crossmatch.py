import numpy as np
from astropy import table
import pandas as pd 
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy.table import hstack
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5
import astropy.units as u


# load in cross-match of cosmos2020 with specz truth elgs 
cosmos_crossmatch_input_name = 'cosmos2020_spec_truth_crossmatch'
cosmos_crossmatch_input = f'{cosmos_crossmatch_input_name}.hdf5'
cosmos_crossmatch_file_path = str(Path(f'../data/processed/hdf5/{cosmos_crossmatch_input}'))
cosmos_truth_cat = read_table_hdf5(input=cosmos_crossmatch_file_path)


# load in full hsc wide cat
hsc_full_input_name = 'hsc_wide_mags_failure_flags'
output = f'{hsc_full_input_name}.hdf5'
hsc_full_path = str(Path(f'../data/processed/hdf5/{output}'))
hsc_full_cat = read_table_hdf5(input=hsc_full_path)

# cross-match the cosmos_truth_cat to the full hsc wide cat
ra_cosmos_truth = cosmos_truth_cat['TARGET_RA']
dec_cosmos_truth = cosmos_truth_cat['TARGET_DEC']
ra_hsc_full = hsc_full_cat['ra']
dec_hsc_full = hsc_full_cat['dec']

cosmos_truth_coord = SkyCoord(ra_cosmos_truth * u.degree, dec_cosmos_truth * u.degree)
hsc_full_coord = SkyCoord(ra_hsc_full * u.degree, dec_hsc_full * u.degree)
idx_pz, d2d_pz, d3d_pz = cosmos_truth_coord.match_to_catalog_sky(hsc_full_coord)
dmask_hsc_pz = d2d_pz.arcsec < 1
cosmos_truth_wide_cat = hstack([cosmos_truth_cat, hsc_full_cat[idx_pz]])[dmask_hsc_pz]

# write to hdf5 file in hsc processed directory
crossmatch_output_name = 'cosmos2020_spec_truth_hsc_wide_crossmatch'
crossmatch_output = f'{crossmatch_output_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_output}'))
write_table_hdf5(cosmos_truth_wide_cat, output=crossmatch_file_path, serialize_meta=True, overwrite=True)


# code to test if you can read in the catalog from hdf5 file:

test_hsc_table = read_table_hdf5(input=crossmatch_file_path)
print(test_hsc_table['Z'])
