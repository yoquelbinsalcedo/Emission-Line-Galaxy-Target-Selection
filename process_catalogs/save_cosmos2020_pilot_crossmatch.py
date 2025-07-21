from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy.table import hstack
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5
import astropy.units as u


# load in catalogs from hdf5 files
cosmos_input_name = 'cosmos2020'
cosmos_input = f'{cosmos_input_name}.hdf5'
cosmos_file_path = str(Path(f'../data/processed/hdf5/{cosmos_input}'))
cosmos_cat = read_table_hdf5(input=cosmos_file_path)

pilot_input_name = 'desi2_pilot'
pilot_input = f'{pilot_input_name}.hdf5'
pilot_file_path = str(Path(f'../data/processed/hdf5/{pilot_input}'))
pilot_cat = read_table_hdf5(input=pilot_file_path)

# cross-match with HSC-W for photometry

ra_cosmos = cosmos_cat['RA']
dec_cosmos = cosmos_cat['DEC']
ra_pilot = pilot_cat['RA']
dec_pilot = pilot_cat['DEC']

cosmos_coord = SkyCoord(ra_cosmos, dec_cosmos)
pilot_coord = SkyCoord(ra_pilot*u.degree, dec_pilot*u.degree)
idx_r, d2d_r, d3d_r = pilot_coord.match_to_catalog_sky(cosmos_coord)
dmask_r = d2d_r.arcsec < 1
cosmos_pilot_cat = hstack([pilot_cat, cosmos_cat[idx_r]])[dmask_r]

# write to hdf5 file in hsc processed directory
crossmatch_output_name = 'cosmos2020_pilot_crossmatch'
crossmatch_output = f'{crossmatch_output_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_output}'))
write_table_hdf5(cosmos_pilot_cat, output=crossmatch_file_path, serialize_meta=True, overwrite=True)


# code to test if you can read in the catalog from hdf5 file:

test_hsc_table = read_table_hdf5(input=crossmatch_file_path)
print(test_hsc_table['Z'])
