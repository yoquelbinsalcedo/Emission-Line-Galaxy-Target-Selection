#Imports 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from astropy import units as u 
from astropy import table
from astropy.coordinates import SkyCoord

#load in cosmos2020 catalog
# Loading the Farmer version of the COSMOS2020 Cat
catversion1 = 'Farmer'  
dir_in1 ='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'  
dir_out1 = '/Users/yokisalcedo/Desktop/data/' # the directory where the output of this notebook will be stored
cat = table.Table.read(dir_in1+'COSMOS2020_{}_jan_processed.fits'.format(catversion1),format='fits',hdu=1).to_pandas()



#load in HSC catalog
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
# Upload the main catalogue
hsc_cat = table.Table.read(dir_in+'HSC.fits',format='fits',hdu=1).to_pandas()

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

#Cross match HSC and COSMOS2020
hsc_ra = hsc_cat['ra']
hsc_dec = hsc_cat['dec']
ra = cat['RA']
dec = cat['DEC']
c_hsc = SkyCoord(ra = hsc_ra.values*u.degree, dec=hsc_dec.values*u.degree)
c_cosmos = SkyCoord(ra=ra.values*u.degree, dec=dec.values*u.degree)
idx, d2d, d3d = c_cosmos.match_to_catalog_sky(c_hsc)
idx 
d2d.arcsec
d3d
dmask = d2d.arcsec < 1 
hsc_cat = hsc_cat.reset_index()
match_hsc = hsc_cat.iloc[idx]
combined_cat = pd.concat([cat, match_hsc.reset_index()], axis = 1)[dmask]

#Mask to get rid of Nans and cut off g/r 
colormasky = np.logical_and.reduce((np.isfinite(combined_cat['HSC_r_MAG'].values),
                                    np.isfinite(combined_cat['photoz'].values),
                                    np.isfinite(combined_cat['HSC_g_MAG'].values), np.isfinite(combined_cat['HSC_i_MAG'].values),
                                    np.isfinite(combined_cat['HSC_y_MAG'].values), np.isfinite(combined_cat['CFHT_u_MAG'].values),
                                    np.isfinite(combined_cat['HSC_z_MAG'].values),
                                    np.isfinite(combined_cat['HSC_z_MAG'].values),
                                    (np.logical_or(combined_cat['HSC_g_MAG'].values < 24.5, combined_cat['HSC_r_MAG'].values < 24.5))))

#Define photoz and colors with HSC photometry after crossmatching
photoz = combined_cat['photoz'][colormasky]
gmag_comb = combined_cat['g_mag'][colormasky]
rmag_comb = combined_cat['r_mag'][colormasky]
imag_comb = combined_cat['i_mag'][colormasky]
ymag_comb = combined_cat['y_mag'][colormasky]
zmag_comb = combined_cat['z_mag'][colormasky]
gfib_comb = combined_cat['g_fiber_mag'][colormasky]
rfib_comb = combined_cat['r_fiber_mag'][colormasky]
iz = imag_comb - zmag_comb
ri = rmag_comb - imag_comb
iy = imag_comb - ymag_comb

#Final sample for spectra cuts
gfib_mask = gfib_comb < 24.3
gband = np.logical_and(gmag_comb < 24, gfib_mask)
rfib_mask = gfib_comb < 24.3
rband = np.logical_and(rmag_comb < 24, rfib_mask)
inlimit = np.logical_or(rband, gband)
cuts_iyri = np.logical_and((iy > 0.35),(ri < iy - 0.19)) 
cuts_final = np.logical_and(cuts_iyri, inlimit)

#Plot Histogram of photoz distribution for our sample to obtain spectra for
fig, ax = plt.subplots()
ax.hist(photoz[cuts_final], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'black')
ax.axvline(x = 1.05,ls= '--', color='black')
ax.axvline(x = 1.55,ls= '--', color='black')
ax.set_xlabel('Lephare Photometric Redshift', fontsize = 18)
ax.set_ylabel('Count', fontsize = 18)
plt.tick_params(axis = 'both', labelsize = 12)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/targetdist.png', dpi = 300, bbox_inches='tight' )