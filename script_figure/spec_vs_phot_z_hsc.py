#imports
import numpy as np
from astropy import units as u 
from astropy import table
import astropy.units as u
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import hstack

def plot_and_stats(z_spec,z_phot, ylabel, isvert = True, vlinemin = 1.05, vlinemax = 1.65, filepath = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/spec_vs_phot_z_hsc'):
    
    x = np.arange(0,5.4,0.05)

# define differences of >0.15*(1+z) as non-Gaussian 'outliers'    
    outlier_upper = x + 0.15*(1+x)
    outlier_lower = x - 0.15*(1+x)

    mask = np.abs((z_phot - z_spec)/(1 + z_spec)) > 0.15
    notmask = ~mask 
    
#Standard Deviation of the predicted redshifts compared to the data:
    std_result = np.std((z_phot - z_spec)/(1 + z_spec), ddof=1)

#Normalized MAD (Median Absolute Deviation):
    nmad = 1.48 * np.median(np.abs((z_phot - z_spec)/(1 + z_spec)))

#Percentage of delta-z > 0.15(1+z) outliers:
    eta = np.sum(np.abs((z_phot - z_spec)/(1 + z_spec)) > 0.15)/len(z_spec)
    
    #Median offset (normalized by (1+z); i.e., bias:
    bias = np.median(((z_phot - z_spec)/(1 + z_spec)))

    sigbias=std_result/np.sqrt(0.64*len(z_phot))
    
     # make photo-z/spec-z plot
    if isvert:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 15))
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5)) 
    
    #add lines to indicate outliers
    ax0.plot(x, outlier_upper, 'k--')
    ax0.plot(x, outlier_lower, 'k--')
    ax0.plot(z_spec[mask], z_phot[mask], 'r.', markersize=6,  alpha=0.5)
    ax0.plot(z_spec[notmask], z_phot[notmask], 'b.',  markersize=6, alpha=0.5)
    ax0.plot(x, x, linewidth=1.5, color = 'red')
    ax0.set_title('$\sigma_\mathrm{NMAD} \ = $%6.4f\n'%nmad+'$(\Delta z)>0.15(1+z) $ outliers = %6.3f'%(eta*100)+'%', fontsize=18)
    ax0.set_xlim(0, 2)
    ax0.set_ylim(0, 2)
    ax0.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 27)
    ax0.set_ylabel(ylabel, fontsize = 27)
    ax0.xaxis.set_tick_params(labelsize = 16)
    ax0.yaxis.set_tick_params(labelsize = 16)
    #ax1 for the histogram
    ax1.hist(z_spec, bins = np.linspace(0, 2, 100), histtype= 'step', color = 'black', label = 'specz')
    ax1.hist(z_phot, bins = np.linspace(0, 2, 100), histtype= 'step', color = 'red', label = 'photoz')
    ax1.axvline(x = vlinemin,ls= '--', color='black')
    ax1.axvline(x = vlinemax,ls= '--', color='black')
    ax1.set_xlabel('Redshift', fontsize = 27)
    ax1.set_ylabel('Count', fontsize = 27)
    ax1.legend(loc = 'upper left', fontsize = 16)
    ax1.xaxis.set_tick_params(labelsize = 16)
    ax1.yaxis.set_tick_params(labelsize = 16)
    plt.savefig(filepath, dpi = 300, bbox_inches='tight')
    plt.show()
    

#Load in catalogs 

#load in specz cataloges 
tert = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/all_elgs.fits',format='fits',hdu=1)
#cleaning specz catalog
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0 
exposure = tert['TSNR2_LRG']*12.15
tmask = exposure > 200
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, tert['YSH']))
elgs = tert[t_mask]

#Load in hsc catalog with grz band photozs
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
# Upload the main catalogue
hsc_cat_pz = table.Table.read(dir_in+'hsc_cat_pz.fits',format='fits',hdu=1)

#merge both hsc with photozs and elgs catalogs 
ra_elg = elgs['TARGET_RA']
dec_elg = elgs['TARGET_DEC']
ra_hsc_pz = hsc_cat_pz['ra']
dec_hsc_pz = hsc_cat_pz['dec']
hsc_pz_coord = SkyCoord(ra_hsc_pz*u.degree, dec_hsc_pz*u.degree)
elg_coord = SkyCoord(ra_elg*u.degree, dec_elg*u.degree)
idx_pz, d2d_pz, d3d_pz = elg_coord.match_to_catalog_sky(hsc_pz_coord)
dmask_hsc_pz = d2d_pz.arcsec <  0.000001
combined_cat_hsc_pz = hstack([elgs, hsc_cat_pz[idx_pz]])[dmask_hsc_pz]

o2_snr_comb_pz = combined_cat_hsc_pz['OII_FLUX']*np.sqrt( combined_cat_hsc_pz['OII_FLUX_IVAR'])   
chi2_comb_pz =  combined_cat_hsc_pz['DELTACHI2']
snr_mask_comb_pz = o2_snr_comb_pz > 10**(0.9 - 0.2*np.log10(chi2_comb_pz))
snr_mask_comb_pz = np.logical_or(snr_mask_comb_pz, chi2_comb_pz > 25)

#photz vs. specz using hsc photoz
plot_and_stats(combined_cat_hsc_pz['Z'][snr_mask_comb_pz], combined_cat_hsc_pz['photoz_best'][snr_mask_comb_pz], ylabel = 'HSC photoz', isvert = True)

