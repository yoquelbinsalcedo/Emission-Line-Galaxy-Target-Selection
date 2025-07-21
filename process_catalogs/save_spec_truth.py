import numpy as np
from astropy import table
import pandas as pd 
from pathlib import Path
import h5py
from astropy.io.misc.hdf5 import write_table_hdf5
from astropy.io.misc.hdf5 import read_table_hdf5

# load in hsc-fits photometry and spectra (exposure, O II SNR, etc) for the extended ELG DESI program targets Rongpu and his team explored
dir_in_rongpu = Path('../data/input/elgs_desi2_truth_hsc_matched.csv')
hsc_elgs_rongpu_all_exp = pd.read_csv(dir_in_rongpu)
exposure_mask = hsc_elgs_rongpu_all_exp['EFFTIME'] > 700
columns = {"Z_x": "Z", "DELTACHI2_x": "DELTACHI2"
           }
hsc_elgs_truth = hsc_elgs_rongpu_all_exp.rename(columns=columns)[exposure_mask]
hsc_elgs_truth = table.Table.from_pandas(hsc_elgs_truth)

print(len(hsc_elgs_truth))


# write to hdf5 file in hsc processed directory

output_name = 'spec_truth'
output = f'{output_name}.hdf5'
hdf5_file_path = str(Path(f'../data/processed/hdf5/{output}'))
write_table_hdf5(hsc_elgs_truth, output=hdf5_file_path, serialize_meta=True, overwrite=True)


# code to test if you can read in the catalog from hdf5 file:

test_hsc_table = read_table_hdf5(input=hdf5_file_path)
print(test_hsc_table['Z'])
