import numpy as np
from astropy import units as u
from astropy import table
from astropy.table import hstack
from astropy.coordinates import SkyCoord
from pathlib import Path
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5

# Load in hsc catalog with grz band photozs
dir_in_hsc = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/input/hsc_cat_pz.fits'
hsc_cat_pz = table.Table.read(dir_in_hsc, format='fits', hdu=1)

# load in the full hsc catalog
hsc_full_input_name = 'hsc_wide_mags_failure_flags'
output = f'{hsc_full_input_name}.hdf5'
hsc_full_path = str(Path(f'../data/processed/hdf5/{output}'))
hsc_cat_full = read_table_hdf5(input=hsc_full_path)

# cross-match to the full hsc catalog so we can use photometry we used for optimized color cuts and g-fiber limiting mag

ra_hsc_pz = hsc_cat_pz['ra']
dec_hsc_pz = hsc_cat_pz['dec']
ra_hsc_full = hsc_cat_full['ra']
dec_hsc_full = hsc_cat_full['dec']

hsc_pz_coord = SkyCoord(ra_hsc_pz * u.degree, dec_hsc_pz * u.degree)
hsc_full_coord = SkyCoord(ra_hsc_full * u.degree, dec_hsc_full * u.degree)
idx_pz, d2d_pz, d3d_pz = hsc_pz_coord.match_to_catalog_sky(hsc_full_coord)
dmask_hsc_pz = d2d_pz.arcsec < 1
combined_cat_hsc = hstack([hsc_cat_pz, hsc_cat_full[idx_pz]])[dmask_hsc_pz]




# write to hdf5 file
output_name = 'hsc_wide_photz'
output = f'{output_name}.hdf5'
hdf5_file_path = str(Path(f'../data/processed/hdf5/{output}'))
write_table_hdf5(combined_cat_hsc, output=hdf5_file_path, serialize_meta=True, overwrite=True)


# code to test if you can read in the catalog from hdf5 file:
test_hsc_table = read_table_hdf5(input=hdf5_file_path)
print(test_hsc_table['g_mag'])
