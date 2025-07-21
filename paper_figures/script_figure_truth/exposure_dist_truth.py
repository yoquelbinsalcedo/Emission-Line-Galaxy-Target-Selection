import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

truth_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_wide_crossmatch.hdf5'
truth_cat = read_table_hdf5(input=truth_file_path)

# histogram showing the distribution of exposure times for objects that are elgs, all without reliable redshifts cuts applied.
fig, ax0 = plt.subplots()

ax0.hist(truth_cat['EFFTIME'], bins=5000, color='red')
ax0.set_xlim(0, 3500)
ax0.set_ylim(0, 10000)
ax0.axvline(x = 1400, c='black')
ax0.axvline(x = 700, c='black')
ax0.xaxis.set_tick_params(labelsize = 20)
ax0.yaxis.set_tick_params(labelsize = 20)
ax0.set_xlabel('Effective Exposure Time (s)', fontsize=30)
ax0.set_ylabel('Count', fontsize=30)
# xfill_truth = np.linspace(700, 1400, 100)
# yfill1_truth = np.linspace(50000, 50000, 100)
# yfill2_truth = np.linspace(0, 0, 100)
# ax0.fill_between(xfill_truth, yfill1_truth, yfill2_truth, color='grey', alpha=0.5)

plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/exposure_dist_truth.png', dpi=300, bbox_inches='tight' )