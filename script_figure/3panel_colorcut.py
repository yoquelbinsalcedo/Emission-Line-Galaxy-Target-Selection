
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from astropy import units as u 
from astropy import table
import pickle
from astropy.coordinates import SkyCoord
import colorcet as cc

#load in cosmos2020 catalog
catversion1 = 'Farmer'  
dir_in1 ='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'  
dir_out1 = '/Users/yokisalcedo/Desktop/data/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
cat = table.Table.read(dir_in1+'COSMOS2020_{}_jan_processed.fits'.format(catversion1),format='fits',hdu=1).to_pandas()
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

#load in HSC catalogs 
dir_in = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'   
dir_out = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/' # the directory where the output of this notebook will be stored
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

#Fitting RF to combined catalog
colormasky = np.logical_and.reduce((np.isfinite(combined_cat['HSC_r_MAG'].values),
                                    np.isfinite(combined_cat['photoz'].values),
                                    np.isfinite(combined_cat['HSC_g_MAG'].values), np.isfinite(combined_cat['HSC_i_MAG'].values),
                                    np.isfinite(combined_cat['HSC_y_MAG'].values), np.isfinite(combined_cat['CFHT_u_MAG'].values),
                                    np.isfinite(combined_cat['HSC_z_MAG'].values),
                                    np.isfinite(combined_cat['HSC_z_MAG'].values),
                                    (np.logical_or(combined_cat['HSC_g_MAG'].values < 24.5, combined_cat['HSC_r_MAG'].values < 24.5))))


#Define what to fit with RF and define hsc mags
photoz = combined_cat['photoz'][colormasky]
xallb_comb = combined_cat.loc[colormasky,['ug','ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
gmag_comb = combined_cat['g_mag'][colormasky]
rmag_comb = combined_cat['r_mag'][colormasky]
imag_comb = combined_cat['i_mag'][colormasky]
ymag_comb = combined_cat['y_mag'][colormasky]
zmag_comb = combined_cat['z_mag'][colormasky]
gfib_comb = combined_cat['g_fiber_mag'][colormasky]
rfib_comb = combined_cat['r_fiber_mag'][colormasky]

#Load in pickled RF fit using all color and mag features
pickled_clfallb_comb = pickle.load(open('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/clfallb_comb.pkl', 'rb'))
proballb_comb = pickled_clfallb_comb.predict_proba(xallb_comb)
probmaskallb_comb = proballb_comb[:, 1] >= 0.025

#Define colors to plot
iz = imag_comb - zmag_comb
ri = rmag_comb - imag_comb
iy = imag_comb - ymag_comb

#Color cuts
cuts = np.logical_and(iy - 0.19  > ri, iy > 0.35)
not_color =~ cuts
cuts_iz = np.logical_and.reduce((iy - 0.19 > ri, iy > 0.35, iz > 0.374))
cuts_iz_only = iz > 0.374
cuts_rf = proballb_comb[:, 1] >= 0.025

#Color coding by the average photoz value per pixel
avg_photoz_ri_iy = np.nanmean(photoz[cuts_iz])
avg_photoz_ri_iz = np.nanmean(photoz[cuts_iz])
avg_photoz_iy_iz = np.nanmean(photoz[cuts_iz])

#bins for all colors
bin_ri = np.linspace(-0.5,1.5,21)
bin_iy = np.linspace(0,1.5,21)
bin_iz = np.linspace(-0.15,1.0,21)

#masking i-y vs r-i with rf and i-z cut
hist7, _, _ = np.histogram2d(ri[cuts_iz],iy[cuts_iz],bins=[bin_ri,bin_iy])
hist8,_,_ = np.histogram2d(ri[cuts_rf & cuts_iz],iy[cuts_rf & cuts_iz],bins=[bin_ri,bin_iy])
frac_ri_iy_cut = hist8/hist7

#masking i-z vs r-i with rf and ri/iy cut
hist9, _,_ = np.histogram2d(ri[cuts_iz],iz[cuts_iz],bins=[bin_ri,bin_iz])
hist10,_,_ = np.histogram2d(ri[cuts_rf & cuts_iz],iz[cuts_rf & cuts_iz],bins=[bin_ri,bin_iz])
frac_ri_iz_cut = hist10/hist9

#masking i-z vs i-y with rf and ri/iy cut
hist11, _,_ = np.histogram2d(iy[cuts_iz],iz[cuts_iz],bins=[bin_iy,bin_iz])
hist12, _,_ = np.histogram2d(iy[cuts_rf & cuts_iz],iz[cuts_rf & cuts_iz],bins=[bin_iy,bin_iz])
frac_iy_iz_cut = hist12/hist11

# Create empty arrays to store mean photoz values
mean_photoz_ri_iy = np.full_like(frac_ri_iy_cut, np.nan)
mean_photoz_ri_iz = np.full_like(frac_ri_iz_cut, np.nan)
mean_photoz_iy_iz = np.full_like(frac_iy_iz_cut, np.nan)

# Loop over each pixel and calculate mean photoz value
for i in range(len(bin_ri)-1):
    for j in range(len(bin_iy)-1):
        mask_ri_iy= np.logical_and.reduce((ri[cuts_iz] >= bin_ri[i], ri[cuts_iz] < bin_ri[i+1], iy[cuts_iz] >= bin_iy[j], iy[cuts_iz] < bin_iy[j+1]))
        mean_photoz_ri_iy[j, i] = np.nanmean(photoz[cuts_iz][mask_ri_iy])

for i in range(len(bin_ri)-1):
    for j in range(len(bin_iz)-1):
        mask_ri_iz = np.logical_and.reduce((ri[cuts_iz] >= bin_ri[i], ri[cuts_iz] < bin_ri[i+1], iz[cuts_iz] >= bin_iz[j], iz[cuts_iz] < bin_iz[j+1]))
        mean_photoz_ri_iz[j, i] = np.nanmean(photoz[cuts_iz][mask_ri_iz])

for i in range(len(bin_iy)-1):
    for j in range(len(bin_iz)-1):
        mask_iy_iz = np.logical_and.reduce((iy[cuts_iz] >= bin_iy[i], iy[cuts_iz] < bin_iy[i+1], iz[cuts_iz] >= bin_iz[j], iz[cuts_iz] < bin_iz[j+1]))
        mean_photoz_iy_iz[j, i] = np.nanmean(photoz[cuts_iz][mask_iy_iz])

#3 panel of fraction of objects passing our RF prob => 0.025 cut after applying color cut in r,i,z,y color space
#colormap to use
cmap = cc.cm.kbc 
cmap.set_extremes(under = 'red', over = 'springgreen') 

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12,8), 
                                                       #subplot_kw=dict(aspect='equal'),
                                                       constrained_layout = True,
                                                       )
vmin = 1.1
vmax = 1.6
kwargs = dict(cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
sd0 = ax0.imshow(mean_photoz_ri_iy, extent=(-0.5, 1.5, 0, 1.5), aspect=2/1.5, **kwargs)
sd1 = ax1.imshow(mean_photoz_ri_iz, extent=(-0.5, 1.5, -0.15, 1.0), aspect=2/1.15, **kwargs)
sd2 = ax2.imshow(mean_photoz_iy_iz, extent=(0, 1.5, -0.15, 1.0), aspect=1.5/1.15, **kwargs)

# #after color cuts
ri_lim = (-0.5,1.5)
iy_lim = (0,1.5)
iz_lim = (-0.15, 1.0)

ax0.set_xlabel('r-i', fontsize = 18)
ax0.set_ylabel('i-y', fontsize = 18)
ax0.axhline(y = 0.35, xmax = 0.33, c = 'black') 
ax0.set_xlim(*ri_lim)
ax0.set_ylim(*iy_lim)
x = np.arange(0.35, 2, .05)
ax0.plot(x - 0.19 ,x , c = 'black')
ax0.xaxis.set_tick_params(labelsize = 12)
ax0.yaxis.set_tick_params(labelsize = 12)

ax1.set_xlabel('r-i', fontsize = 18)
ax1.set_ylabel('i-z', fontsize = 18)
ax1.xaxis.set_tick_params(labelsize = 12)
ax1.yaxis.set_tick_params(labelsize = 12)
ax1.axhline(y = 0.374, ls = '--', c = 'black') 
ax1.set_xlim(*ri_lim)
ax1.set_ylim(*iz_lim)

ax2.set_xlabel('i-y', fontsize = 18)
ax2.set_ylabel('i-z', fontsize = 18)
ax2.xaxis.set_tick_params(labelsize = 12)
ax2.yaxis.set_tick_params(labelsize = 12)
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
ax2.axhline(y = 0.374, ls = '--', c = 'black') 
ax2.set_xlim(*iy_lim)
ax2.set_ylim(*iz_lim)
cbar = plt.colorbar(sd2, ax=[ax0, ax1, ax2], orientation='horizontal',  extend = 'both', shrink=0.75)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Average photoz', fontsize=15, loc='center')
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/3panel_colorcut.png', dpi = 300, bbox_inches='tight' )
