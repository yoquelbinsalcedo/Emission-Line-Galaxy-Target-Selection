import numpy as np
from matplotlib import pyplot as plt
from astropy import table
import pandas as pd
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5


pilot_input_name = 'desi2_pilot'
pilot_input = f'{pilot_input_name}.hdf5'
pilot_file_path = str(Path(f'../data/processed/hdf5/{pilot_input}'))
pilot_cat = read_table_hdf5(input=pilot_file_path)
pilot_exposure = pilot_cat['TSNR2_LRG']*12.15

# load in spectroscopic truth sample

truth_input_name = 'spec_truth'
truth_input = f'{truth_input_name}.hdf5'
truth_file_path = str(Path(f'../data/processed/hdf5/{truth_input}'))
truth_cat = read_table_hdf5(input=truth_file_path)

# histogram showing the distribution of exposure times for objects that are elgs, all without reliable redshifts cuts applied.
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12,20), sharex=True)

ax0.hist(truth_cat['EFFTIME'], bins=5000, color='red')
ax0.set_title('Spectroscopic truth', fontsize=27)
ax0.set_xlim(0, 3500)
ax0.set_ylim(0, 10000)
ax0.axvline(x = 1400, c='black')
ax0.axvline(x = 700, c='black')
ax0.xaxis.set_tick_params(labelsize = 19)
ax0.yaxis.set_tick_params(labelsize = 19)
xfill_truth = np.linspace(700, 1400, 100)
yfill1_truth = np.linspace(50000, 50000, 100)
yfill2_truth = np.linspace(0, 0, 100)
ax0.fill_between(xfill_truth, yfill1_truth, yfill2_truth, color='grey', alpha=0.5)

ax1.hist(pilot_exposure, bins=400, color='blue')
ax1.set_title('DESI-2 pilot', fontsize=27)
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 1500)
ax1.set_ylabel('Count', fontsize=25)
ax1.axvline(x = 1400, c = 'black')
ax1.axvline(x = 200, c = 'black')
ax1.xaxis.set_tick_params(labelsize = 19)
ax1.yaxis.set_tick_params(labelsize = 19)
ax1.set_xlabel('Exposure Time (s)', fontsize=25)
xfill_yoki = np.linspace(200, 1400,100)
yfill1_yoki = np.linspace(1500, 1500,100)
yfill2_yoki = np.linspace(0, 0, 100)
ax1.fill_between(xfill_yoki, yfill1_yoki, yfill2_yoki, color='grey', alpha=0.5)

plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/exposure_dist_truth_yoki.png', dpi=300, bbox_inches='tight' )