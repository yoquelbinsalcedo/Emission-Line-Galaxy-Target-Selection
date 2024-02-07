#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u 
from astropy import table
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord
import colorcet as cc


# load in catalogs
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


#load in specz catalogs
tert = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/all_elgs.fits',format='fits',hdu=1)
#cleaning specz catalog
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0 
exposure = tert['TSNR2_LRG']*12.15
tmask = exposure > 200
ysh = tert['YSH'] == True
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, ysh ))
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

#Specz combined catalog
specz = combined_cat['Z'][snr_mask_comb]

#Best case for 1.05 < z <  1.65 for rioffset, iyoffset, izmin, and gfiblim values
rishift = 0 
iyshift = 0.06636818841724967
izmin = 0.36442580263398095
gfiblim = 24.263228615646897

color_mask = np.logical_and((ri < iy - 0.19 + rishift),
                             (iy > 0.35 + iyshift))
color_mask &= iz > izmin
gfibcut = combined_cat['g_fiber_mag'][snr_mask_comb] < gfiblim
ccuts = np.logical_and(color_mask, gfibcut) 

#note that the plot below is only illustrating the final color cuts, not including the gfiber cut.

#colormap to use
cmap = cc.cm.kbc 
cmap.set_extremes(under = 'red', over = 'springgreen') 

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(24, 10), 
                                                       #subplot_kw=dict(aspect='equal'),
                                                       #constrained_layout = True,
                                                       )
vmin = 1.05
vmax = 1.65

sd0 = ax0.scatter(ri, iy, c = specz , cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri, iz, c = specz, cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy, iz, c = specz , cmap = cmap, s = 12, alpha = 0.1 , vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x = 100, y = 100, c = combined_cat['Z'][0], cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)

# #after color cuts
ri_lim = (-0.5,1.5)
iy_lim = (0,1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize = 27)
ax0.set_ylabel('i-y', fontsize = 27)
ax0.axhline(y = 0.35, xmax = 0.33, ls = '--', c = 'black') 
ax0.axhline(y = 0.35 + iyshift , xmax = 0.360 , c = 'black')  
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
x = np.arange(0.35, 2, .05)
y = x - 0.19
ax0.plot(y ,x , ls = '--', c = 'black')
x_new = np.arange(0.41 , 2, .05)
y_new = x_new - 0.19
ax0.plot(y_new ,x_new, c = 'black') 
ax0.xaxis.set_tick_params(labelsize = 16)
ax0.yaxis.set_tick_params(labelsize = 16)

ax1.set_xlabel('r-i', fontsize = 27)
ax1.set_ylabel('i-z', fontsize = 27)
ax1.xaxis.set_tick_params(labelsize = 16)
ax1.yaxis.set_tick_params(labelsize = 16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)
ax1.axhline(y = izmin , color='black')

ax2.set_xlabel('i-y', fontsize = 27)
ax2.set_ylabel('i-z', fontsize = 27)
ax2.xaxis.set_tick_params(labelsize = 16)
ax2.yaxis.set_tick_params(labelsize = 16)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.axhline(y = izmin , color='black')
ax2.axvline(x = 0.35, ls = '--', color='black')
ax2.axvline(x = 0.35 + iyshift , color='black')
cbar = plt.colorbar(sdummy, ax=[ax0, ax1, ax2], orientation='horizontal',  extend = 'both', shrink = 0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Spectroscopic redshift', fontsize = 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/3panel_color_old_new_scat.png', dpi = 300, bbox_inches='tight' )
