#imports
import numpy as np
import colorcet as cc
from matplotlib import pyplot as plt
from astropy.table import Table 
from astropy import units as u 
from astropy import table

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


#Load in specz cataloges 
tert = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/all_elgs.fits',format='fits',hdu=1)
#Cleaning specz catalog
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0 
exposure = tert['TSNR2_LRG']*12.15
tmask = exposure > 200
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, tert['YSH'] == True))
elgs = tert[t_mask]
#Criteria for good redshifts
o2_snr = elgs['OII_FLUX']*np.sqrt(elgs['OII_FLUX_IVAR'])   
chi2 = elgs['DELTACHI2']
snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
snr_mask = np.logical_or(snr_mask, chi2 > 25)
x = np.linspace(-1,5,100)
y = (0.9 - 0.2*(x))

#Colors for color coding
cmap = cc.cm.kbc 
cmap.set_extremes(under = 'red', over = 'springgreen')

#Make scatter plot of objects that pass and fail snr_mask
fig, ax = plt.subplots()
ax.scatter(np.log10(chi2), np.log10(o2_snr), c = elgs['Z'], cmap = cmap, s = 2, alpha = 0.1, vmin= 1, vmax= 1.65)
ax.plot(x, y, color = 'black', label = 'y = 0.9 - 0.2 * x')
cdummy = ax.scatter(x = 100, y = 100, c = elgs['Z'][0], cmap = cmap, s = 5, alpha = 1 ,vmin= 1, vmax= 1.65)
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 2)
ax.set_xlabel('log10($\Delta\chi^2$)', fontsize = 18)
ax.set_ylabel('log10([OII]SNR)', fontsize = 18)
ax.xaxis.set_tick_params(labelsize = 12)
ax.yaxis.set_tick_params(labelsize = 12)
ax.axvline(1.40, ls = '--', label = '$\Delta\chi^2$ = 25', c = 'black')
cbar = plt.colorbar(cdummy, extend = 'both')
cbar.set_label('spectroscopic redshift', fontsize = 18)
ax.legend(fontsize = 13)
xfill = np.linspace(-1, 1.40,2)
yfill1 = np.linspace(1.1, 0.6200000000000001,2)
yfill2 = np.linspace(-1,-1, 2)
ax.fill_between(xfill,yfill1,yfill2, color = 'grey', alpha = 0.5)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/specz_citeria_scatt.png', dpi = 300, bbox_inches='tight')
