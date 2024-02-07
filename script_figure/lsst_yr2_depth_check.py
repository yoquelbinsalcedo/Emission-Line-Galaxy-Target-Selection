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

# merge both hsc_cat and elgs catalogs, this combined catalog will be used to tweak the cuts and check our redshift distribution
ra_elg = elgs['TARGET_RA']
dec_elg = elgs['TARGET_DEC']
ra_hsc = hsc_cat['ra']
dec_hsc = hsc_cat['dec']
hsc_coord = SkyCoord(ra_hsc*u.degree, dec_hsc*u.degree)
elg_coord = SkyCoord(ra_elg*u.degree, dec_elg*u.degree)
idx_h, d2d_h, d3d_h = elg_coord.match_to_catalog_sky(hsc_coord)
dmask = d2d_h.arcsec < 0.000001
combined_cat = hstack([elgs, hsc_cat[idx_h]])[dmask]

# masking with snr and chi2 cuts for combined_cat
o2_snr_comb = combined_cat['OII_FLUX']*np.sqrt(combined_cat['OII_FLUX_IVAR'])   
chi2_comb = combined_cat['DELTACHI2']
snr_mask_comb = o2_snr_comb > 10**(0.9 - 0.2*np.log10(chi2_comb))
snr_mask_comb = np.logical_or(snr_mask_comb, chi2_comb > 25)
snr_mask_comb_fail =~ snr_mask_comb

# mags
g = combined_cat['g_mag'][snr_mask_comb]
r = combined_cat['r_mag'][snr_mask_comb]
i = combined_cat['i_mag'][snr_mask_comb]
z = combined_cat['z_mag'][snr_mask_comb]
y = combined_cat['y_mag'][snr_mask_comb]

#colors
ri = r - i
iy = i - y
iz = i - z

# Best case for 1.05 < z <  1.65 for rioffset, iyoffset, izmin, and gfiblim values
rishift = 0 
iyshift = 0.06636818841724967
izmin = 0.36442580263398095
gfiblim = 24.263228615646897

color_mask = np.logical_and(
    (ri < iy - 0.19 + rishift),
    (iy > 0.35 + iyshift))
color_mask &= iz > izmin


#check if we are close to LSST year 2 coadded 5sigma depths
# u : 23.8, 25.6
# g : 24.5, 26.9
# r : 24.03, 26.9
# i : 23.41 , 26.4
# z : 22.74, 25.6
# y : 22.96, 24.8
sig5_gr = (26.9 + 2.5*np.log10(2/10)) #LSST year 2 depth r band 
sig5_i = (26.4 + 2.5*np.log10(2/10)) #LSST year 2 depth i band 
sig5_z = (25.6 + 2.5*np.log10(2/10)) #LSST year 2 depth z band 
sig5_y = (24.8 + 2.5*np.log10(2/10)) #LSST year 2 depth y band 

sig5_depths = (sig5_gr, sig5_i, sig5_z, sig5_y)
mags = (g, r, i, z, y)

# # plot all 5 mags and their year 2 5sigma depths as a vertical line 
# fig, ax = plt.subplots(figsize=(12,8))
# ax.hist(r[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='red', label='r')
# ax.axvline(sig5_gr, color='red', linestyle='--', lw = 3, label='r 5$\sigma$ depth')
# ax.hist(i[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='blue', label='i')
# ax.axvline(sig5_i, color='blue', linestyle='--', lw = 3, label='i 5$\sigma$ depth')
# ax.hist(z[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='purple', label='z')
# ax.axvline(sig5_z, color='purple', linestyle='--', lw = 3, label='z 5$\sigma$ depth')   
# ax.hist(y[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='orange', label='y')
# ax.axvline(sig5_y, color='orange', linestyle='--', lw = 3, label='y 5$\sigma$ depth')
# plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/lsst_yr2_depth_check.png')   

# now make individual subplots plots for each band of the exact same thing stacked veritcally
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, ncols=1, figsize=(10,35),  sharex='all')
ax0.hist(g[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='green', label='g')
ax0.text(21,300, s='g', fontsize = 30, color = 'green')
ax0.set_title('LSST year 2 5$\sigma$ depths', fontsize=25)
ax1.hist(r[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='red', label='r')
ax1.text(21,300, s='r', fontsize = 30, color = 'red')
ax2.hist(i[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='blue', label='i')
ax2.text(21,300, s='i', fontsize = 30, color = 'blue')
ax3.hist(z[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='purple', label='z')
ax3.text(21,300, s='z', fontsize = 30, color = 'purple')
ax4.hist(y[color_mask], bins = np.linspace(18, 28, 150), histtype='step', color='orange', label='y')
ax4.text(21,300, s='y', fontsize = 30, color = 'orange')

ax0.axvline(sig5_gr, color='green', linestyle='--', lw = 3)
ax1.axvline(sig5_gr, color='red', linestyle='--', lw = 3)
ax2.axvline(sig5_i, color='blue', linestyle='--', lw = 3)
ax3.axvline(sig5_z, color='purple', linestyle='--', lw = 3)
ax4.axvline(sig5_y, color='orange', linestyle='--', lw = 3)
for ax in (ax0, ax1, ax2, ax3, ax4):
    #ax.set_box_aspect(1)
    ax.set_xlim(21,26)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    #ax.legend(loc='upper left', fontsize=29)

plt.xlabel('Apparent magnitude', fontsize=45)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/lsst_yr2_depth_check_vert.png', bbox_inches='tight', dpi=300)  

