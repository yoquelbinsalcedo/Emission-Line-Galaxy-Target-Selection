import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u 
from astropy import table
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord

#functions:
#A function that takes an hsc_cat as input (as well as shifts and magnitude limits) and output surface densities
def surf_density(cat_full, rishift = 0, iyshift = 0, izmin = -99,  glim = 99, gfiblim = 99):
    '''This function takes in a catalog and the color cuts to use and outputs the surface density'''
    #define the color cuts
    rishift_min = np.minimum(0,rishift)
    cuts_iyri = np.logical_and((cat_full['r_mag'] - cat_full['i_mag'] < cat_full['i_mag'] - cat_full['y_mag'] - 0.19 + rishift_min),
                           (cat_full['i_mag'] - cat_full['y_mag'] > 0.35 + iyshift)) 
    colorcuts = np.logical_and(cuts_iyri, cat_full['i_mag'] - cat_full['z_mag'] > izmin)
    
    #define the magnitude cuts
    gfib_mask = cat_full['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(cat_full['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final = np.logical_and(colorcuts, gbandcut)
    cutcat = cat_full[cuts_final]
    #calculate the surface density
    area = 16
    surf_density = len(cutcat)/area
    return surf_density


#Function with the following as inputs
#the catalog to work with
#how much to shift the cutoffs
#the magnitude limit to use for g (default: 99)
#the magnitude limit to use for gfiber (default: 99)
#it would output: the success rate and redshift range success rate.

def success_rate(catalog, rishift = 0, iyshift = 0, izmin = -99,  glim = 99, gfiblim = 99):
    '''This function takes in a catalog and the color cuts to use and outputs the success rate and redshift range success rate'''
    #define the color cuts
    cuts_iyri = np.logical_and((catalog['r_mag'] - catalog['i_mag'] < catalog['i_mag'] - catalog['y_mag'] - 0.19 + rishift),
                           (catalog['i_mag'] - catalog['y_mag'] > 0.35 + iyshift)) 
    colorcuts = np.logical_and(cuts_iyri, catalog['i_mag'] - catalog['z_mag'] > izmin)
    #exposure cuts
    exposure = catalog['TSNR2_LRG']*12.15
    tmask = exposure < 1400
    #define the magnitude cuts
    gfib_mask = catalog['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(catalog['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final_200 = np.logical_and(colorcuts, gbandcut)
    cuts_final = np.logical_and(cuts_final_200, tmask)
    cutcat_200 = catalog[cuts_final_200]
    cutcat = catalog[cuts_final]
    #calculate the success rate
    o2_snr = cutcat['OII_FLUX']*np.sqrt(cutcat['OII_FLUX_IVAR'])   
    chi2 = cutcat['DELTACHI2']
    o2_snr_200 = catalog['OII_FLUX'][cuts_final_200]*np.sqrt(catalog['OII_FLUX_IVAR'][cuts_final_200])   
    chi2_200 = catalog['DELTACHI2'][cuts_final_200]
    #masking specz catalog
    snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
    snr_mask = np.logical_or(snr_mask, chi2 > 25)
    snr_mask_200 = o2_snr_200 > 10**(0.9 - 0.2*np.log10(chi2_200))
    snr_mask_200 = np.logical_or(snr_mask_200, chi2_200 > 25)
    #calculate the fraction of objects with good redshifts with the 200 < t < 1400 sample 
    success_rate_200_1400 = np.sum(snr_mask)/np.sum(cuts_final)
    #calculate the redshift range success rate with 200 < t sample
    redshift_range_200 = np.sum(np.logical_and(snr_mask_200, np.logical_and(cutcat_200['Z'] > 1.05, cutcat_200['Z'] < 1.65)))/np.sum(cuts_final_200)
    #Calculate the net redshift yield
    redshift_range = redshift_range_200*success_rate_200_1400
    return success_rate_200_1400, redshift_range

def success_rate_6(catalog, rishift = 0, iyshift = 0, izmin = -99,  glim = 99, gfiblim = 99):
    '''This function takes in a catalog and the color cuts to use and outputs the success rate and redshift range success rate'''
    #define the color cuts
    cuts_iyri = np.logical_and((catalog['r_mag'] - catalog['i_mag'] < catalog['i_mag'] - catalog['y_mag'] - 0.19 + rishift),
                           (catalog['i_mag'] - catalog['y_mag'] > 0.35 + iyshift)) 
    colorcuts = np.logical_and(cuts_iyri, catalog['i_mag'] - catalog['z_mag'] > izmin)
    #exposure cuts
    exposure = catalog['TSNR2_LRG']*12.15
    tmask = exposure < 1400
    #define the magnitude cuts
    gfib_mask = catalog['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(catalog['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final_200 = np.logical_and(colorcuts, gbandcut)
    cuts_final = np.logical_and(cuts_final_200, tmask)
    cutcat_200 = catalog[cuts_final_200]
    cutcat = catalog[cuts_final]
    #calculate the success rate
    o2_snr = cutcat['OII_FLUX']*np.sqrt(cutcat['OII_FLUX_IVAR'])   
    chi2 = cutcat['DELTACHI2']
    o2_snr_200 = catalog['OII_FLUX'][cuts_final_200]*np.sqrt(catalog['OII_FLUX_IVAR'][cuts_final_200])   
    chi2_200 = catalog['DELTACHI2'][cuts_final_200]
    #masking specz catalog
    snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
    snr_mask = np.logical_or(snr_mask, chi2 > 25)
    snr_mask_200 = o2_snr_200 > 10**(0.9 - 0.2*np.log10(chi2_200))
    snr_mask_200 = np.logical_or(snr_mask_200, chi2_200 > 25)
    #calculate the fraction of objects with good redshifts with the 200 < t < 1400 sample 
    success_rate_200_1400 = np.sum(snr_mask)/np.sum(cuts_final)
    #calculate the redshift range success rate with 200 < t sample
    redshift_range_200 = np.sum(np.logical_and(snr_mask_200, np.logical_and(cutcat_200['Z'] > 1.1, cutcat_200['Z'] < 1.6)))/np.sum(cuts_final_200)
    #Calculate the net redshift yield
    redshift_range = redshift_range_200*success_rate_200_1400
    return success_rate_200_1400, redshift_range

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

#load in specz cataloges 
tert = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/all_elgs.fits',format='fits',hdu=1)
#cleaning specz catalog
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0 
exposure = tert['TSNR2_LRG']*12.15
tmask = exposure > 200
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, tert['YSH']))
elgs = tert[t_mask]

#merge both hsc_cat and elgs catalogs, this combined catalog will be used to tweak the cuts and check our redshift distribution
ra_elg = elgs['TARGET_RA']
dec_elg = elgs['TARGET_DEC']
ra_hsc = hsc_cat['ra']
dec_hsc = hsc_cat['dec']
hsc_coord = SkyCoord(ra_hsc*u.degree, dec_hsc*u.degree)
elg_coord = SkyCoord(ra_elg*u.degree, dec_elg*u.degree)
idx_h, d2d_h, d3d_h = elg_coord.match_to_catalog_sky(hsc_coord)
dmask = d2d_h.arcsec < 0.000001
combined_cat = hstack([elgs, hsc_cat[idx_h]])[dmask]

#Tweak the final gfib limiting magnitude with a goal of 1370 objs/deg^2 for 1.05 < z < 1.65
nmag = 60
maglim= np.linspace(23,24.5,nmag)
gfib_lim = 24.215

zsuccessf = np.zeros(nmag)
rangesuccessf = np.zeros(nmag)
surface_densityf = np.zeros(nmag)
surf_density_actf = np.zeros(nmag)
for i, mlim in enumerate(maglim):
      ratesf = success_rate(combined_cat,rishift = 0,iyshift = 0.05636818841724967,izmin = 0.37442580263398095,gfiblim = mlim)
      surff = surf_density(hsc_cat, rishift = 0,iyshift = 0.05636818841724967,izmin = 0.37442580263398095,gfiblim = mlim)
      zsuccessf[i] = ratesf[0]
      rangesuccessf[i] = ratesf[1]
      surface_densityf[i] = surff
      surf_density_actf[i] = surff*rangesuccessf[i]

# Best case for 1.05 < z <  1.65 for rioffset, iyoffset, izmin, and gfiblim values
# rishift = 0, iyshift = 0.05636818841724967, izmin = 0.37442580263398095 , gfiblim = 24.253228615646897
density_fin = surf_density(hsc_cat, rishift = 0, iyshift = 0.05636818841724967, izmin = 0.37442580263398095 , gfiblim = 24.253228615646897)
zsuccess_fin, rangesuccess_fin = success_rate(combined_cat, rishift = 0, iyshift = 0.05636818841724967, izmin = 0.37442580263398095 , gfiblim = 24.253228615646897)
print('for rishift = 0 , iyshift = 0.05636818841724967, izmin = 0.37442580263398095, and gfiblim = 24.253228615646897, z success/ z range success are', zsuccess_fin, rangesuccess_fin, 'target density of', density_fin,'and surface density yield is', density_fin*rangesuccess_fin)

# plot of surface density vs lim gfib mag with optimal values of rishift, iysift, and izmin from the scipy minimizer
fig, ax = plt.subplots()
ax.plot(maglim, surface_densityf, label = f'Optimized Cuts', color = 'black')
ax.axhline(y=1930,ls= ':',color='orange', label = 'DESI ELG LOP ')
ax.axvline(x=24.25,ls= '--',color='blue', label = 'DESI-2 ELGs ')
ax.axvline(x=24.1,ls= ':',color='orange')
ax.set_xlabel('Limiting g Fiber Magnitude', fontsize = 17)
ax.set_ylabel(r'Surface Density ($\frac{objs}{deg^2}$)', fontsize = 17)
ax.xaxis.set_tick_params(labelsize = 12)
ax.yaxis.set_tick_params(labelsize = 12)
ax.legend(loc = 'upper left', fontsize = 14)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/targdensity_limgfibmag', dpi = 300, bbox_inches='tight')