import scipy as sp
import time
import numpy as np 
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt 
from astropy.io import fits,ascii,votable
from astropy import units as u 
from astropy import constants as const
from astropy import table
from astropy.table import join
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from astropy.coordinates import SkyCoord
from pathlib import Path
from astropy.io.misc.hdf5 import read_table_hdf5

def cmap_white(cmap_name):
    """Returns a colormap with white as the lowest value color."""
    import numpy as np
    try:
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        cmap = cm.get_cmap(cmap_name, 256)
    except ValueError:
        import seaborn as sns
        cmap = sns.color_palette("flare", as_cmap=True)
    newcolors = cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 0])
    newcolors[:1, :] = white
    cmap_white = ListedColormap(newcolors)
    return cmap_white


crossmatch_output_name = 'hsc_wide_cosmos2020_crossmatch'
crossmatch_output = f'{crossmatch_output_name}.hdf5'
crossmatch_file_path = str(Path(f'../data/processed/hdf5/{crossmatch_output}'))
combined_cat = read_table_hdf5(input=crossmatch_file_path).to_pandas()


zallb_comb = combined_cat['photoz']
xallb_comb = combined_cat[['ug', 'ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
gmag_comb = combined_cat['g_mag']
rmag_comb = combined_cat['r_mag']
imag_comb = combined_cat['i_mag']
ymag_comb = combined_cat['y_mag']
zmag_comb = combined_cat['z_mag']
gfib_comb = combined_cat['g_fiber_mag']
rfib_comb = combined_cat['r_fiber_mag']
# xallb_comb_train, xallb_comb_test, gfib_comb_train, gfib_comb_test, rfib_comb_train, rfib_comb_test, gmag_comb_train, gmag_comb_test, rmag_comb_train, rmag_comb_test, imag_comb_train, imag_comb_test, zmag_comb_train, zmag_comb_test, ymag_comb_train, yallb_comb_test, yb_comb_train, yb_comb_test, zallb_comb_train, zallb_comb_test = train_test_split(xallb_comb, gfib_comb, rfib_comb, gmag_comb, rmag_comb, imag_comb, zmag_comb, ymag_comb, yb_comb, zmag_comb)

pickled_clfallb_comb = pickle.load(open('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/models/clfallb_comb.pkl', 'rb'))
proballb_comb = pickled_clfallb_comb.predict_proba(xallb_comb)
probmaskallb_comb = proballb_comb[:, 1] >= 0.025


# define colors to plot
iz = imag_comb - zmag_comb
ri = rmag_comb - imag_comb
iy = imag_comb - ymag_comb

# color cuts
rishift, iyshift, izmin, gfiblim = 0, 5.979861E-02, 3.634577E-01, 2.422445E+01

cuts = np.logical_and(iy - 0.19  > ri, iy > 0.35)
not_color =~ cuts
cuts_iz = np.logical_and.reduce((iy - 0.19 > ri, iy > 0.35, iz > 0.374))
cuts_iz_only = iz > 0.374
cuts_rf = proballb_comb[:, 1] >= 0.025
cuts_rf50 = proballb_comb[:, 1] >= 0.50
not_rf =~ cuts_rf

# objects that did not meet color cuts but have RF_prob > 0.025
not_color_rf = np.logical_and(not_color, cuts_rf )

# final sample for spectra cuts
gfib_mask = gfib_comb < 24.3
gband = np.logical_and(gmag_comb < 24, gfib_mask)
rfib_mask = gfib_comb < 24.3
rband = np.logical_and(rmag_comb < 24, rfib_mask)
inlimit = np.logical_or(rband, gband)
cuts_iyri = np.logical_and((iy > 0.35),(ri < iy - 0.19)) 
cuts_final = np.logical_and(cuts_iyri, inlimit)

#photoz
photoz = zallb_comb

#bins for all colors
bin_ri = np.linspace(-0.5,1.5,101)
bin_iy = np.linspace(0,1.5,101)
bin_iz = np.linspace(-0.15,1.0,101)

# fraction of objects that pass rf prob in i-y >= 0.025 in i-y vs r-i color space 
hist1, _,_ = np.histogram2d(ri,iy,bins=[bin_ri,bin_iy])
hist2,_,_ = np.histogram2d(ri[cuts_rf],iy[cuts_rf],bins=[bin_ri,bin_iy])
frac_ri_iy = hist2/hist1

#fraction of objects that pass rf prob in i-y >= 0.025 in i-z vs r-i color space 
hist3, _,_ = np.histogram2d(ri,iz,bins=[bin_ri,bin_iz])
hist4, _,_ = np.histogram2d(ri[cuts_rf],iz[cuts_rf],bins=[bin_ri,bin_iz])
frac_ri_iz = hist4/hist3

#fraction of objects that pass rf prob in i-y >= 0.025 in i-z vs i-y color space 
hist5, _,_ = np.histogram2d(iy,iz,bins=[bin_iy,bin_iz])
hist6, _,_ = np.histogram2d(iy[cuts_rf],iz[cuts_rf],bins=[bin_iy,bin_iz])
frac_iy_iz = hist6/hist5

#masking i-y vs r-i with rf and i-z cut
hist7, _, _ = np.histogram2d(ri[cuts],iy[cuts],bins=[bin_ri,bin_iy])
hist8,_,_ = np.histogram2d(ri[cuts_rf & cuts],iy[cuts_rf & cuts],bins=[bin_ri,bin_iy])
frac_ri_iy_izcut = hist8/hist7

#masking i-z vs r-i with rf and ri/iy cut
hist9, _,_ = np.histogram2d(ri[cuts],iz[cuts],bins=[bin_ri,bin_iz])
hist10,_,_ = np.histogram2d(ri[cuts_rf & cuts],iz[cuts_rf & cuts],bins=[bin_ri,bin_iz])
frac_ri_iz_cut = hist10/hist9

#masking i-z vs i-y with rf and ri/iy cut
hist11, _,_ = np.histogram2d(iy[cuts],iz[cuts],bins=[bin_iy,bin_iz])
hist12, _,_ = np.histogram2d(iy[cuts_rf & cuts],iz[cuts_rf & cuts],bins=[bin_iy,bin_iz])
frac_iy_iz_cut = hist12/hist11

#colormap
cmap = cmap_white("viridis")

# 2 x 6 panel of i-z vs r-i and i-z vs i-y before and after r-i < i-y -0.19 cut and i-y vs r-i before and after i-z color cut applied with fraction => 0.025
    
fig, ([ax0, ax1, ax2], [ax3, ax4, ax5]) = plt.subplots(nrows=2, ncols=3, figsize=(12,8), 
                                                       # subplot_kw=dict(aspect='equal'),
                                                       constrained_layout = True,
                                                       )
#before color cuts
sd0 = ax0.imshow(frac_ri_iy.T,cmap=cmap, origin = 'lower', extent = (-0.5, 1.5, 0, 1.5), aspect = 2/1.5)
sd1 = ax1.imshow(frac_ri_iz.T,cmap=cmap, origin = 'lower', extent = (-0.5, 1.5, -0.15, 1.0), aspect = 2/1.15)
sd2 = ax2.imshow(frac_iy_iz.T,cmap=cmap, origin = 'lower', extent = (0, 1.5, -0.15, 1.0), aspect = 1.5/1.15)
# #after color cuts
sd3 = ax3.imshow(frac_ri_iy_izcut.T,cmap=cmap, origin = 'lower',extent = (-0.5, 1.5, 0, 1.5), aspect=2/1.5)
sd4 = ax4.imshow(frac_ri_iz_cut.T,cmap=cmap, origin = 'lower', extent = (-0.5, 1.5, -0.15, 1.0),  aspect = 2/1.15)
sd5 = ax5.imshow(frac_iy_iz_cut.T,cmap=cmap, origin = 'lower', extent = (0, 1.5, -0.15, 1.0), aspect = 1.5/1.15)
ri_lim = (-0.5,1.5)
iy_lim = (0,1.5)
iz_lim = (-0.15, 1.0)
ax0.set_xlabel('r-i', fontsize = 20)
ax0.set_ylabel('i-y', fontsize = 20)
#ax0.axhline(y = 0.35, xmax = 0.33, c = 'red') 
x = np.arange(0.35, 2, .05)
#ax0.plot(x - 0.19 ,x , c = 'red')
ax0.xaxis.set_tick_params(labelsize = 12)
ax0.yaxis.set_tick_params(labelsize = 12)
ax1.set_xlabel('r-i', fontsize = 20)
ax1.set_ylabel('i-z', fontsize = 20)
ax1.xaxis.set_tick_params(labelsize = 12)
ax1.yaxis.set_tick_params(labelsize = 12)
ax2.set_xlabel('i-y', fontsize = 20)
ax2.set_ylabel('i-z', fontsize = 20)
ax2.xaxis.set_tick_params(labelsize = 12)
ax2.yaxis.set_tick_params(labelsize = 12)
cbar = fig.colorbar(sd2, ax= [ax2,ax5], location='right')
cbar.set_label('Fraction passing random forest cut', fontsize = 18, loc = 'center')
#make color bar tick labels bigger
cbar.ax.tick_params(labelsize=18)
ax3.axhline(y = 0.35, xmax = 0.33, c = 'red') 
ax3.plot(x - 0.19 ,x , c = 'red')
ax3.set_xlabel('r-i', fontsize = 20)
ax3.set_ylabel('i-y', fontsize = 20)
ax3.xaxis.set_tick_params(labelsize = 12)
ax3.yaxis.set_tick_params(labelsize = 12)
ax4.set_xlabel('r-i', fontsize = 20)
ax4.set_ylabel('i-z', fontsize = 20)
ax4.xaxis.set_tick_params(labelsize = 12)
ax4.yaxis.set_tick_params(labelsize = 12)
ax5.set_xlabel('i-y', fontsize = 20)
ax5.set_ylabel('i-z', fontsize = 20)
ax5.xaxis.set_tick_params(labelsize = 12)
ax5.yaxis.set_tick_params(labelsize = 12)
ax5.axvline(x = 0.35, ymin = -0.15, ymax = 1.0, ls = '-', c = 'red')
ax0.set_title('before color cut',fontsize = 20)
ax1.set_title('before color cut',fontsize = 20)
ax2.set_title('before color cut ',fontsize = 20)
ax3.set_title('after color cut',fontsize = 20)
ax4.set_title('after color cut',fontsize = 20)
ax5.set_title('after color cut',fontsize = 20)
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax3.set_xlim(*ri_lim)
ax3.set_ylim(*iy_lim)
ax4.set_xlim(*ri_lim)
ax4.set_ylim(*iz_lim)
ax5.set_xlim(*iy_lim)
ax5.set_ylim(*iz_lim) 

plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_pilot/colors_rf_frac_2by6.png', dpi = 300, bbox_inches = 'tight') 
