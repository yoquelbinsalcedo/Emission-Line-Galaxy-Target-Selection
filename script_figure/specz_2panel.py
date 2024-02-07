#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table 
from astropy import units as u 
from astropy import table
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord

#load in catalogs 
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
dir_out = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
hsc_cat = table.Table.read(dir_in+'HSC.fits',format='fits',hdu=1)

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

#Merge both hsc_cat and elgs catalogs, this combined catalog will be used to tweak the cuts and check our redshift distribution
ra_hsc = hsc_cat['ra']
dec_hsc = hsc_cat['dec']
ra_elg = elgs['TARGET_RA']
dec_elg = elgs['TARGET_DEC']
hsc_coord = SkyCoord(ra_hsc*u.degree, dec_hsc*u.degree)
elg_coord = SkyCoord(ra_elg*u.degree, dec_elg*u.degree)
idx_h, d2d_h, d3d_h = elg_coord.match_to_catalog_sky(hsc_coord)
dmask = d2d_h.arcsec < 0.000001
combined_cat = hstack([elgs, hsc_cat[idx_h]])[dmask]

#Good redshift criteria for combined_cat
o2_snr_comb = combined_cat['OII_FLUX']*np.sqrt(combined_cat['OII_FLUX_IVAR'])   
chi2_comb = combined_cat['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail =~ snr_mask_comb

#Specz before color optimization and combined catalog specz to mask with final color cuts
specz_before = elgs['Z'][snr_mask]
specz_comb = combined_cat['Z'][snr_mask_comb]

#Color cuts and gfib lim after optimization for 1.05 < z < 1.65 sample
rishift = 0 
iyshift = 0.06636818841724967
izmin = 0.36442580263398095
gfiblim = 24.263228615646897

color_mask = np.logical_and((combined_cat['r_mag'][snr_mask_comb] - combined_cat['i_mag'][snr_mask_comb] < combined_cat['i_mag'][snr_mask_comb] - combined_cat['y_mag'][snr_mask_comb] - 0.19 + rishift),
                             (combined_cat['i_mag'][snr_mask_comb] - combined_cat['y_mag'][snr_mask_comb] > 0.35 + iyshift))
color_mask &= (combined_cat['i_mag'][snr_mask_comb] - combined_cat['z_mag'][snr_mask_comb]) > izmin
ccuts = np.logical_and(color_mask, combined_cat['g_fiber_mag'][snr_mask_comb] < gfiblim) 

#Plot the 2 histograms side by side comparing our sample before and after optimizing color cuts and gfiber limiting mag
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.hist(specz_before, bins = np.linspace(0, 2, 100), histtype= 'step', color = 'black', label = 'Before color cut optimization')
ax1.axvline(x = 1.05,ls= '--', color='black')
ax1.axvline(x = 1.65,ls= '--', color='black')
ax1.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 18)
ax1.set_ylabel('Count', fontsize = 18)
ax1.xaxis.set_tick_params(labelsize = 12)
ax1.yaxis.set_tick_params(labelsize = 12)

ax2.hist(specz_comb[ccuts], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'black', label = 'After color cut optimization')
ax2.axvline(x = 1.05,ls= '--', color='black')
ax2.axvline(x = 1.65,ls= '--', color='black')
ax2.xaxis.set_tick_params(labelsize = 12)
ax2.yaxis.set_tick_params(labelsize = 12)
ax2.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 18)
ax2.set_ylabel('Count', fontsize = 18)
ax1.legend(loc = 'upper left', fontsize = 16)
ax2.legend(loc = 'upper left', fontsize = 16)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/specz_2panel.png', dpi = 300, bbox_inches='tight' )
