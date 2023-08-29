#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table 
from astropy import units as u 
from astropy import table
import astropy.units as u

#load in catalogs 
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
dir_out = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
hsc_cat = table.Table.read(dir_in+'HSC.fits',format='fits',hdu=1)

#function to convert flux to mags
def flux_to_mag(flux):
    return -2.5*np.log10(flux*1e-9) + 8.90
# extinction corrected mags (extinction is negligible for XMM-LSS)
hsc_cat["i_mag"] = flux_to_mag(hsc_cat["i_cmodel_flux"])-hsc_cat["a_i"]
hsc_cat["r_mag"] = flux_to_mag(hsc_cat["r_cmodel_flux"])-hsc_cat["a_r"]
hsc_cat["z_mag"] = flux_to_mag(hsc_cat["z_cmodel_flux"])-hsc_cat["a_z"]
hsc_cat["g_mag"] = flux_to_mag(hsc_cat["g_cmodel_flux"])-hsc_cat["a_g"]
hsc_cat["y_mag"] = flux_to_mag(hsc_cat["y_cmodel_flux"])-hsc_cat["a_y"]
hsc_cat["i_fiber_mag"] = flux_to_mag(hsc_cat["i_fiber_flux"])-hsc_cat["a_i"]
hsc_cat["i_fiber_tot_mag"] = flux_to_mag(hsc_cat["i_fiber_tot_flux"])-hsc_cat["a_i"]
hsc_cat["g_fiber_mag"] = flux_to_mag(hsc_cat["g_fiber_flux"])-hsc_cat["a_g"]
hsc_cat["g_fiber_tot_mag"] = flux_to_mag(hsc_cat["g_fiber_tot_flux"])-hsc_cat["a_g"]
hsc_cat["r_fiber_mag"] = flux_to_mag(hsc_cat["r_fiber_flux"])-hsc_cat["a_r"]
hsc_cat["r_fiber_tot_mag"] = flux_to_mag(hsc_cat["r_fiber_tot_flux"])-hsc_cat["a_r"]
    


## Quality cuts
# valid I-band flux
mask = np.isfinite(hsc_cat["i_cmodel_flux"]) & (hsc_cat["i_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["i_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["i_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["g_cmodel_flux"]) & (hsc_cat["g_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["g_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["g_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["r_cmodel_flux"]) & (hsc_cat["r_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["r_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["r_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["y_cmodel_flux"]) & (hsc_cat["y_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["y_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["y_sdsscentroid_flag"])
mask &= np.isfinite(hsc_cat["z_cmodel_flux"]) & (hsc_cat["z_cmodel_flux"]>0)
#cmodel fit not failed
mask &= (~hsc_cat["z_cmodel_flag"])
#General Failure Flag
mask &= (~hsc_cat["z_sdsscentroid_flag"])
hsc_cat = hsc_cat[mask]

'''grey would have color cuts but no mag cuts, colors would have color and mag cuts'''
#Color cuts used to get spectra 
cuts = np.logical_and(hsc_cat['i_mag'] - hsc_cat['y_mag'] - 0.19 > hsc_cat['r_mag'] - hsc_cat['i_mag'], hsc_cat['i_mag'] - hsc_cat['y_mag'] > 0.35)
gmag_color_cut = np.logical_and(hsc_cat['g_mag'] < 24, hsc_cat['g_fiber_mag'] < 24.3)
gmag_color_cut = np.logical_and(cuts, gmag_color_cut )
rmag_color_cut = np.logical_and(hsc_cat['r_mag'] < 24, hsc_cat['r_fiber_mag'] < 24.3)
rmag_color_cut = np.logical_and(cuts, rmag_color_cut )
inlimit =  np.logical_or(gmag_color_cut, rmag_color_cut)



#4 panel plot of g mag/gfiber & r mag/rfiber with lines showing our cuts used for spectra 
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,figsize=(12,8), ncols=2,  constrained_layout = True)
ax1.hist(hsc_cat['g_mag'][cuts], bins = np.linspace(18, 28, 150), color = 'grey')
ax1.hist(hsc_cat['g_mag'][inlimit], bins = np.linspace(18, 28, 150), color = 'blue')
ax1.set_xlabel('g magnitude ', fontsize = 18)
ax1.axvline(x = 24,ls= '--', color='black')
# ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))

ax2.hist(hsc_cat['g_fiber_mag'][cuts], bins = np.linspace(18, 28, 150), color = 'grey')
ax2.hist(hsc_cat['g_fiber_mag'][inlimit], bins = np.linspace(18, 28, 150), color = 'blue')
ax2.set_xlabel('g Fiber Magnitude', fontsize = 18)
ax2.axvline(x = 24.3,ls= '--', color='black')
# ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))

ax3.hist(hsc_cat['r_mag'][cuts], bins = np.linspace(18, 28, 150), color = 'grey')
ax3.hist(hsc_cat['r_mag'][inlimit], bins = np.linspace(18, 28, 150), color = 'blue')
ax3.set_xlabel('r magnitude ', fontsize = 18)
ax3.axvline(x = 24,ls= '--', color='black')
# ax3.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))

ax4.hist(hsc_cat['r_fiber_mag'][cuts], bins = np.linspace(18, 28, 150), color = 'grey')
ax4.hist(hsc_cat['r_fiber_mag'][inlimit], bins = np.linspace(18, 28, 150), color = 'blue')
ax4.set_xlabel('r Fiber Magnitude', fontsize = 18)
ax4.axvline(x = 24.3,ls= '--', color='black')
# ax4.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0,0))


for ax in (ax1, ax2, ax3, ax4):
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xlim(20, 27)
    ax.set_ylabel('Count * 10^4', fontsize = 18)
    yticklabels = [f'{it / 1e4:0.1f}' for it in ax.get_yticks()]
    ax.set_yticks(ticks=ax.get_yticks(), labels=yticklabels)
    ax.set_ylim(0, 27000)

plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/magdist.png', dpi = 300, bbox_inches='tight' )