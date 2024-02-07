#Imports 
import scipy as sp
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from astropy import units as u 
from astropy import table
import pickle
from sklearn.model_selection import train_test_split
from astropy.coordinates import SkyCoord

#load in cosmos2020 catalog
# Loading the Farmer version of the COSMOS2020 Cat
catversion1 = 'Farmer'  
dir_in1 ='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/'  
dir_out1 = '/Users/yokisalcedo/Desktop/data/' # the directory where the output of this notebook will be stored
# Upload the main catalogue
cat = table.Table.read(dir_in1+'COSMOS2020_{}_jan_processed.fits'.format(catversion1),format='fits',hdu=1).to_pandas()
print(len(cat))

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


zcutb_comb = np.logical_and(combined_cat['photoz'] > 1.05, combined_cat['photoz'] < 1.55)
zallb_comb = combined_cat['photoz'][colormasky]
xallb_comb = combined_cat.loc[colormasky,['ug','ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
yb_comb = zcutb_comb[colormasky]
gmag_comb = combined_cat['g_mag'][colormasky]
rmag_comb = combined_cat['r_mag'][colormasky]
imag_comb = combined_cat['i_mag'][colormasky]
ymag_comb = combined_cat['y_mag'][colormasky]
zmag_comb = combined_cat['z_mag'][colormasky]
gfib_comb = combined_cat['g_fiber_mag'][colormasky]
rfib_comb = combined_cat['r_fiber_mag'][colormasky]
xallb_comb_train, xallb_comb_test, gfib_comb_train, gfib_comb_test, rfib_comb_train, rfib_comb_test, gmag_comb_train, gmag_comb_test, rmag_comb_train, rmag_comb_test, imag_comb_train, imag_comb_test, zmag_comb_train, zmag_comb_test, ymag_comb_train, yallb_comb_test, yb_comb_train, yb_comb_test, zallb_comb_train, zallb_comb_test = train_test_split(xallb_comb, gfib_comb, rfib_comb, gmag_comb, rmag_comb, imag_comb, zmag_comb, ymag_comb, yb_comb, zmag_comb)
#clfallb_comb = RandomForestClassifier()

# pickle.dump(clfallb_comb, open('clfallb_comb.pkl', 'wb'))
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
cuts_rf50 = proballb_comb[:, 1] >= 0.50
not_rf =~ cuts_rf
#Ojects that did not meet color cuts but have RF_prob > 0.025
not_color_rf = np.logical_and(not_color, cuts_rf )

#Final sample for spectra cuts
gfib_mask = gfib_comb < 24.3
gband = np.logical_and(gmag_comb < 24, gfib_mask)
rfib_mask = gfib_comb < 24.3
rband = np.logical_and(rmag_comb < 24, rfib_mask)
inlimit = np.logical_or(rband, gband)
cuts_iyri = np.logical_and((iy > 0.35),(ri < iy - 0.19)) 
cuts_final = np.logical_and(cuts_iyri, inlimit)

#photoz 
photoz = zallb_comb
print(len(photoz[cuts_rf]))

#Plot the redshift distribution for the different cuts and save
fig, ax = plt.subplots()
ax.hist(photoz[cuts], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'lightgrey', fill = True, label = 'i-y/r-i cut')
ax.hist(photoz[cuts_iz], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'lightblue', fill = True ,label = '+ i-z cut')
ax.hist(photoz[cuts_rf], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'red', label = 'RF prob > 0.025')
ax.axvline(x = 1.05,ls= '--', color='black')
ax.axvline(x = 1.55,ls= '--', color='black')
ax.set_xlabel('LePhare Photometric Redshift', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.xaxis.set_tick_params(labelsize = 12)
ax.yaxis.set_tick_params(labelsize = 12)
ax.legend(loc = 'upper left', fontsize = 16)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/hist_overplots_crossmatch.png', dpi = 300, bbox_inches='tight')