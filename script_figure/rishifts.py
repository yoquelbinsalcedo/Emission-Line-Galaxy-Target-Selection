#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u 
from astropy import table
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord

def thresbins_ri(data,ri,
                  iy,vlinemin,
                  vlinemax,
                  filepath = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/rishifts.png',
                  xlabel = '$z_{\mathrm{spec}}$',ylabel = 'Count',histtype = 'step',xlim=(0, 2),**kwargs):
    fig, ax = plt.subplots( figsize=(8, 6))
    threshold = np.arange(0, 0.20, 0.05)
    ax.set_xlim(xlim)
    ax.xaxis.set_tick_params(labelsize = 12)
    ax.yaxis.set_tick_params(labelsize = 12)
    ax.axvline(vlinemin,ls= '--', color='black')
    ax.axvline(vlinemax,ls= '--', color='black')
    ax.set_xlabel(xlabel, fontsize = 27)
    ax.set_ylabel(ylabel, fontsize = 27)
    ax.xaxis.set_tick_params(labelsize = 16)
    ax.yaxis.set_tick_params(labelsize = 16)
    
    p = []

    for pthres in threshold:
        mask = np.logical_and((iy - 0.19 - pthres > ri) , iy > 0.35)
        _, _, ptemp = ax.hist(data[mask], bins = np.linspace(0, 2, 100) , histtype = histtype, label = f'i-y - 0.19 - {pthres:0.02f} > r-i & i-y > 0.35', **kwargs)
        p.append(ptemp)
        
    
    ax.legend(loc = 'upper left', fontsize = 14)
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')
    plt.show()

        
        
    return fig, ax


#load in catalogs 
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
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
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, tert['YSH'] == True))
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

#Masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat['OII_FLUX']*np.sqrt(combined_cat['OII_FLUX_IVAR'])   
chi2_comb = combined_cat['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail =~ snr_mask_comb

#Colors
ri = combined_cat['r_mag'][snr_mask_comb] -  combined_cat['i_mag'][snr_mask_comb]
iz = combined_cat['i_mag'][snr_mask_comb] -  combined_cat['z_mag'][snr_mask_comb]
iy = combined_cat['i_mag'][snr_mask_comb] -  combined_cat['y_mag'][snr_mask_comb]
gband = combined_cat['g_mag'][snr_mask_comb]
gfiber = combined_cat['g_fiber_mag'][snr_mask_comb]
rband = combined_cat['r_mag'][snr_mask_comb]
rfiber = combined_cat['r_fiber_mag'][snr_mask_comb]

#Specz combined catalog
specz = combined_cat['Z'][snr_mask_comb]

#Specz distribution after tweaking color cuts with changing i-y - 0.19 > r-i cut
thresbins_ri(specz, ri, iy, vlinemin = 1.05, vlinemax = 1.65, xlabel = '$z_{\mathrm{spec}}$')
