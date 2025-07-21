import numpy as np
from astropy import table
import pandas as pd 
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy.table import hstack
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5
import astropy.units as u


# load in catalogs from hdf5 files
hsc_wide_input_name = 'hsc_wide_mags_failure_flags'
hsc_wide_input = f'{hsc_wide_input_name}.hdf5'
hsc_wide_file_path = str(Path(f'../data/processed/hdf5/{hsc_wide_input}'))
hsc_wide_cat = read_table_hdf5(input=hsc_wide_file_path)


def sigma_mag(m_5, xmag):
    '''m_5: LSST year 2 band depth
       x_mag: array of magnitudes of the band you want to calculate sigma for
    '''
    flux_5_f = 10**(((xmag - m_5) / 2.5))
    sigma_f_f = 1/5*flux_5_f
    sigma_m = (2.5/np.log(10))*sigma_f_f
    return sigma_m

# add noise to all hsc bands


bands = ['g', 'r', 'i', 'z', 'y']
five_sig_depth = [26.9, 26.9, 26.4, 25.6, 24.8]  # these are coadded

for band, depth in zip(bands, five_sig_depth):

    year2_depth = (depth + 1.25*np.log10(2/10))  # LSST year 2 depth 
    sigma = sigma_mag(year2_depth, hsc_wide_cat[f'{band}_mag'])
    gauss = np.random.normal(0, 1, len(hsc_wide_cat[f'{band}_mag']))
    hsc_wide_cat[f'{band}_mag_noisy'] = hsc_wide_cat[f'{band}_mag'] + sigma * gauss


truth_input_name = 'spec_truth'
truth_input = f'{truth_input_name}.hdf5'
truth_file_path = str(Path(f'../data/processed/hdf5/{truth_input}'))
truth_cat = read_table_hdf5(input=truth_file_path)


# cross-match with HSC-W for photometry
ra_hsc = hsc_wide_cat['ra']
dec_hsc = hsc_wide_cat['dec']
ra_truth = truth_cat['TARGET_RA']
dec_truth = truth_cat['TARGET_DEC']


hsc_coord = SkyCoord(ra_hsc*u.degree, dec_hsc*u.degree)
truth_coord = SkyCoord(ra_truth*u.degree, dec_truth*u.degree)
idx_r, d2d_r, d3d_r = truth_coord.match_to_catalog_sky(hsc_coord)
dmask_r = d2d_r.arcsec < 1
hsc_wide_truth_cat = hstack([truth_cat, hsc_wide_cat[idx_r]])[dmask_r]

# write to hdf5 file in hsc processed directory

crossmatch_output_name = 'noisy_spec_truth_hsc_wide_crossmatch'
crossmatch_output = f'{crossmatch_output_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_output}'))
write_table_hdf5(hsc_wide_truth_cat, output=crossmatch_file_path, serialize_meta=True, overwrite=True)


# code to test if you can read in the catalog from hdf5 file:
test_hsc_table = read_table_hdf5(input=crossmatch_file_path)
print(test_hsc_table['y_mag_noisy'])
