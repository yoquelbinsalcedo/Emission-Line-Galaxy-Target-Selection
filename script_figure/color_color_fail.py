#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u 
from astropy import table
from astropy.table import hstack
import astropy.units as u
from astropy.coordinates import SkyCoord
import colorcet as cc

#load in cosmos2020 catalog
catversion1 = 'Farmer'  
dir_in1 ='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'  
dir_out1 = '/Users/yokisalcedo/Desktop/data/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
cat = table.Table.read(dir_in1+'COSMOS2020_{}_jan_processed.fits'.format(catversion1),format='fits',hdu=1)

#All possible non-redundant colors are listed below:
#u: ug ,ur ,ui ,uz , uy
#g: gr, gi, gz, gy
#r: ri, rz, ry
#i: iz, iy
#z: zy

ug_cos = cat['CFHT_u_MAG'] - cat['HSC_g_MAG'] 
ur_cos = cat['CFHT_u_MAG'] - cat['HSC_r_MAG']
ui_cos = cat['CFHT_u_MAG'] - cat['HSC_i_MAG']
uz_cos = cat['CFHT_u_MAG'] - cat['HSC_z_MAG'] 
uy_cos = cat['CFHT_u_MAG'] - cat['HSC_y_MAG']
gr_cos = cat['HSC_g_MAG'] - cat['HSC_r_MAG']
gi_cos = cat['HSC_g_MAG'] - cat['HSC_i_MAG']
gz_cos = cat['HSC_g_MAG'] - cat['HSC_z_MAG']
gy_cos = cat['HSC_g_MAG'] - cat['HSC_y_MAG']
ri_cos = cat['HSC_r_MAG'] - cat['HSC_i_MAG']
rz_cos = cat['HSC_r_MAG'] - cat['HSC_z_MAG']
ry_cos = cat['HSC_r_MAG'] - cat['HSC_y_MAG']
iz_cos = cat['HSC_i_MAG'] - cat['HSC_z_MAG']
iy_cos = cat['HSC_i_MAG'] - cat['HSC_y_MAG']
zy_cos = cat['HSC_z_MAG'] - cat['HSC_y_MAG']
r_cos = cat['HSC_r_MAG']
g_cos = cat['HSC_g_MAG']
i_cos = cat['HSC_i_MAG']
y_cos = cat['HSC_y_MAG']
z_cos= cat['HSC_z_MAG']

cat['ug']= ug_cos
cat['ur']= ur_cos
cat['ui']= ui_cos
cat['uz']= uz_cos
cat['uy']= uy_cos
cat['gr']= gr_cos
cat['gi']= gi_cos
cat['gz']= gz_cos
cat['gy']= gy_cos
cat['ri']= ri_cos
cat['rz']= rz_cos
cat['ry']= ry_cos
cat['iz']= iz_cos
cat['iy']= iy_cos
cat['zy']= zy_cos
cat['HSC_r_MAG'] = r_cos
cat['HSC_g_MAG'] = g_cos
cat['HSC_i_MAG'] = i_cos
cat['HSC_y_MAG'] = y_cos
cat['HSC_z_MAG'] = z_cos

colormaskx = np.logical_and.reduce((np.isfinite(cat['photoz']),
                                    np.isfinite(cat['CFHT_u_MAG']),
                                    np.isfinite(cat['HSC_g_MAG']),
                                    np.isfinite(cat['HSC_r_MAG']),
                                    np.isfinite(cat['HSC_i_MAG']),
                                    np.isfinite(cat['HSC_z_MAG']),
                                    np.isfinite(cat['HSC_y_MAG']),
                                    (np.logical_or(cat['HSC_g_MAG']< 24.5, cat['HSC_r_MAG']< 24.5))))
cat_cosmos = cat[colormaskx]

#load in hsc cat
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
combined_cat_hsc_cos = hstack([elgs, hsc_cat[idx_h]])[dmask]

#combine the hsc/elg catalog with cosmos2020 
ra_cat_hsc = combined_cat_hsc_cos['TARGET_RA']
dec_cat_hsc = combined_cat_hsc_cos['TARGET_DEC']
ra_cos = cat_cosmos['RA']
dec_cos = cat_cosmos['DEC']
cat_hsc_coord = SkyCoord(ra_cat_hsc*u.degree, dec_cat_hsc*u.degree)
cos_coord = SkyCoord(ra_cos, dec_cos)
idx_cos_hsc, d2d_cos_hsc, d3d_cos_hsc = cat_hsc_coord.match_to_catalog_sky(cos_coord)
dmask_cos_hsc = d2d_cos_hsc.arcsec < 1
combined_cat_hsc_cos = hstack([combined_cat_hsc_cos, cat_cosmos[idx_cos_hsc]])[dmask_cos_hsc]
print(f'the number of objects in this doubly cross-matched cat is {len(combined_cat_hsc_cos)}')

#Masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat_hsc_cos['OII_FLUX']*np.sqrt(combined_cat_hsc_cos['OII_FLUX_IVAR'])   
chi2_comb = combined_cat_hsc_cos['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail = ~snr_mask_comb

#Colors for objects that did not pass the above snr/chi2 cuts
ri = combined_cat_hsc_cos['r_mag'][snr_mask_comb_fail] -  combined_cat_hsc_cos['i_mag'][snr_mask_comb_fail]
iz = combined_cat_hsc_cos['i_mag'][snr_mask_comb_fail] -  combined_cat_hsc_cos['z_mag'][snr_mask_comb_fail]
iy = combined_cat_hsc_cos['i_mag'][snr_mask_comb_fail] -  combined_cat_hsc_cos['y_mag'][snr_mask_comb_fail]

photoz = combined_cat_hsc_cos['photoz'][snr_mask_comb_fail]

#colormap to use
cmap = cc.cm.kbc 
cmap.set_extremes(under = 'red', over = 'springgreen') 
vmin = 1.05
vmax = 1.65

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize = (24,10)
                                                       #subplot_kw=dict(aspect='equal'),
                                                       #constrained_layout = True,
                                                       )

sd0 = ax0.scatter(ri, iy, c = photoz, cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)
sd1 = ax1.scatter(ri, iz, c = photoz, cmap = cmap, s = 12, alpha = 1, vmin=vmin, vmax=vmax)
sd2 = ax2.scatter(iy, iz, c = photoz, cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)
sdummy = ax1.scatter(x = 100, y = 100, c = photoz[0], cmap = cmap, s = 12, alpha = 1 , vmin=vmin, vmax=vmax)

# #after color cuts
ri_lim = (-0.5,1.5)
iy_lim = (0,1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize = 27)
ax0.set_ylabel('i-y', fontsize = 27)
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
x = np.arange(0.35, 2, .05)
ax0.plot(x - 0.19 ,x , ls = '-', c = 'black')
ax0.axhline(y = 0.35, xmax = 0.33, ls = '-', c = 'black') 
ax0.xaxis.set_tick_params(labelsize = 16)
ax0.yaxis.set_tick_params(labelsize = 16)

ax1.set_xlabel('r-i', fontsize = 27)
ax1.set_ylabel('i-z', fontsize = 27)
ax1.xaxis.set_tick_params(labelsize = 16)
ax1.yaxis.set_tick_params(labelsize = 16)
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)

ax2.set_xlabel('i-y', fontsize = 27)
ax2.set_ylabel('i-z', fontsize = 27)
ax2.xaxis.set_tick_params(labelsize = 16)
ax2.yaxis.set_tick_params(labelsize = 16)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.axvline(x = 0.35, ls = '-', color='black')

cbar = plt.colorbar(sdummy, ax=[ax0, ax1, ax2], orientation='horizontal',  extend = 'both', shrink = 0.75)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Lephare photoz', fontsize= 27, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/color_color_fail.png', dpi = 300, bbox_inches='tight' )



