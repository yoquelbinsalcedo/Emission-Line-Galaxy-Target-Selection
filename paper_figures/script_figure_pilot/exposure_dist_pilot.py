import numpy as np
from matplotlib import pyplot as plt
from astropy import table
import pandas as pd
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5


pilot_input_name = 'desi2_pilot'
pilot_input = f'{pilot_input_name}.hdf5'
pilot_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/desi2_pilot.hdf5'
pilot_cat = read_table_hdf5(input=pilot_file_path)
pilot_exposure = pilot_cat['TSNR2_LRG']*12.15

# histogram showing the distribution of exposure times for objects that are elgs, all without reliable redshifts cuts applied.
fig, ax0 = plt.subplots()
ax0.hist(pilot_exposure, bins=400, color='blue')
ax0.set_xlim(0, 5000)
ax0.set_ylim(0, 1500)
ax0.set_ylabel('Count', fontsize=25)
ax0.axvline(x = 1400, c = 'black')
ax0.axvline(x = 200, c = 'black')
ax0.xaxis.set_tick_params(labelsize = 19)
ax0.yaxis.set_tick_params(labelsize = 19)
ax0.set_xlabel('Effective Exposure Time (s)', fontsize=25)
# xfill_yoki = np.linspace(200, 1400,100)
# yfill1_yoki = np.linspace(1500, 1500,100)
# yfill2_yoki = np.linspace(0, 0, 100)
# ax0.fill_between(xfill_yoki, yfill1_yoki, yfill2_yoki, color='grey', alpha=0.5)

plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_pilot/exposure_dist_pilot.png', dpi=300, bbox_inches='tight' )