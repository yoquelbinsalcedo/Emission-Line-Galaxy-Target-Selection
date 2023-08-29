#Imports 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from astropy import units as u 
from astropy import table
import pickle
from astropy.coordinates import SkyCoord

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

#Fitting RF to combined catalog
colormasky = np.logical_and.reduce((np.isfinite(cat['HSC_r_MAG'].values),
                                    np.isfinite(cat['photoz'].values),
                                    np.isfinite(cat['HSC_g_MAG'].values), np.isfinite(cat['HSC_i_MAG'].values),
                                    np.isfinite(cat['HSC_y_MAG'].values), np.isfinite(cat['CFHT_u_MAG'].values),
                                    np.isfinite(cat['HSC_z_MAG'].values),
                                    (np.logical_or(cat['HSC_g_MAG'].values < 24.5, cat['HSC_r_MAG'].values < 24.5))))


#Define what to fit with RF and define hsc mags
photoz = cat['photoz'][colormasky]
features = cat.loc[colormasky,['ug','ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]
rmag = cat['HSC_r_MAG'][colormasky]
imag = cat['HSC_i_MAG'][colormasky]
ymag = cat['HSC_y_MAG'][colormasky]
zmag = cat['HSC_z_MAG'][colormasky]

#Load in pickled RF fit using all color and mag features
pickled_clf = pickle.load(open('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/clfallb_comb.pkl', 'rb'))
prob = pickled_clf.predict_proba(features)

#Define colors to plot
iz = imag - zmag
ri = rmag - imag
iy = imag - ymag

#Color cuts and random forest cuts
cuts = np.logical_and(iy - 0.19  > ri, iy > 0.35)
probmask = prob[:, 1] >= 0.025

#plot
fig, ax = plt.subplots()
ax.hist(photoz[cuts],bins = np.linspace(0, 2, 100),color = 'grey',histtype = 'step',label = 'Non-degraded y-band selection')
ax.axvline(x = 1.05,ls= '--', color='black')
ax.axvline(x = 1.55,ls= '--', color='black')
ax.xaxis.set_tick_params(labelsize = 12)
ax.yaxis.set_tick_params(labelsize = 12)
ax.set_xlabel('Lephare Photometric Redshift', fontsize = 16)
ax.legend(loc = 'upper left', fontsize = 16)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/select_cosmos_no_yband_noise.png', dpi = 300, bbox_inches='tight' )
