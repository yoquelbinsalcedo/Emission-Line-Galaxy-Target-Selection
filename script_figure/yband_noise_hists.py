 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from astropy import units as u 
from astropy import table
import pickle
from astropy.coordinates import SkyCoord
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def complete_purity_cutsonly(colorcuts, photozs, zcut=[1.05,1.55]):
    
    zrange = np.logical_and(photozs > zcut[0], photozs < zcut[1])
    completeness = np.sum(np.logical_and(colorcuts, zrange)) / np.sum(zrange)
    purity = np.sum(np.logical_and(colorcuts, zrange)) / np.sum(colorcuts)

    return completeness, purity


# load in cosmos2020 catalog
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

#We create an array containing a noisy version of the y magnitudes (m_noisy) by taking the array of the non-noisy y magntiudes (m_initial) 
#and adding the array of sigmas multiplied by an array of equal length consisting of N random numbers distributed as a Gaussian with mean 0 and sigma 1
#We’d want to use LSST year 2 depths to calculate probably final LSST y band depth +2.5*log10(year/10) with year = 2
#LSST final year y band depth are y : 22.96, 24.8 for single image and coadded images
r_cos = cat['HSC_r_MAG']
g_cos = cat['HSC_g_MAG']
i_cos = cat['HSC_i_MAG']
y_cos = cat['HSC_y_MAG']
z_cos = cat['HSC_z_MAG']

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

# function to calculate sigma for each magnitude 
def sigma_mag(m_5, xmag):
    '''m_5: LSST year 2 band depth
       x_mag: array of magnitudes of the band you want to calculate sigma for
    '''
    flux_5_f = 10**(((xmag - m_5) / 2.5))
    sigma_f_f  = 1/5 * flux_5_f
    sigma_m = (2.5/np.log(10))*sigma_f_f
    return sigma_m

mag_5 = (24.8 + 2.5*np.log10(2/10)) # LSST year 2 depth y band 

# y band with noise
sigma_y = sigma_mag(mag_5, y_cos)
gauss = np.random.normal(0, 1, len(y_cos))
y_noise = y_cos + sigma_y*gauss


# define colors based on noisy y band
uy_noisey = cat['CFHT_u_MAG'] - y_noise
gy_noisey = cat['HSC_g_MAG'] - y_noise
ry_noisey = cat['HSC_r_MAG'] - y_noise
iy_noisey = cat['HSC_i_MAG'] - y_noise
zy_noisey = cat['HSC_z_MAG'] - y_noise

cat['HSC_y_MAG_noise'] = y_noise
cat['uy_noise'] = uy_noisey
cat['gy_noise'] = gy_noisey
cat['ry_noise'] = ry_noisey
cat['iy_noise'] = iy_noisey
cat['zy_noise'] = zy_noisey


colormask = np.logical_and.reduce((
    np.isfinite(cat['HSC_r_MAG'].values),
    np.isfinite(cat['photoz'].values),
    np.isfinite(cat['HSC_g_MAG'].values),
    np.isfinite(cat['HSC_i_MAG'].values),
    np.isfinite(cat['HSC_y_MAG'].values), 
    np.isfinite(cat['HSC_y_MAG_noise'].values), 
    np.isfinite(cat['CFHT_u_MAG'].values),
    np.isfinite(cat['HSC_z_MAG'].values),
    (np.logical_or(cat['HSC_g_MAG'].values < 24.5,
                   cat['HSC_r_MAG'].values < 24.5))))


colormask = np.logical_and.reduce((np.isfinite(cat['HSC_r_MAG'].values),
                                    np.isfinite(cat['photoz'].values),
                                    np.isfinite(cat['HSC_g_MAG'].values),
                                      np.isfinite(cat['HSC_i_MAG'].values),
                                    np.isfinite(cat['HSC_y_MAG'].values), 
                                    np.isfinite(cat['HSC_y_MAG_noise'].values), 
                                    np.isfinite(cat['CFHT_u_MAG'].values),
                                    np.isfinite(cat['HSC_z_MAG'].values),
                                    (np.logical_or(cat['HSC_g_MAG'].values < 24.5, cat['HSC_r_MAG'].values < 24.5))))


#Define what to fit with RF and define hsc mags
photoz = cat['photoz'][colormask]
features = cat.loc[colormask,['ug','ur', 'ui', 'uz', 'uy', 'gr', 'gi', 'gz', 'gy', 'ri', 'rz', 'ry', 'iz', 'iy', 'zy','HSC_r_MAG','HSC_i_MAG','HSC_y_MAG','HSC_z_MAG','CFHT_u_MAG', 'HSC_g_MAG']]

y = np.logical_and(photoz > 1.05, photoz < 1.55)
rmag = cat['HSC_r_MAG'][colormask]
imag = cat['HSC_i_MAG'][colormask]
ymag = cat['HSC_y_MAG'][colormask]
ymag_noise = cat['HSC_y_MAG_noise'][colormask]
zmag = cat['HSC_z_MAG'][colormask]

x_train, x_test, photz_train, photz_test, y_train, y_test = train_test_split(features, photoz, y, test_size=0.20)

#train RF
# clf = RandomForestClassifier()
# clf.fit(x_train, y_train)

# # save the model to disk
filename = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/clf_105_155_all_feats.pkl'
# pickle.dump(clf, open(filename, 'wb'))
# load the model from disk
clf = pickle.load(open(filename, 'rb'))

prob = clf.predict_proba(features)[:,1]
prob_mask = prob > 0.025
y_pred = clf.predict(x_test)
auc_score = roc_auc_score(y_test, y_pred)
print('the auc score of the RF with no noise in HSC y mag is', auc_score)


iz = iz_cos[colormask]
ri = ri_cos[colormask]
iy = iy_cos[colormask]
iy_noise = iy_noisey[colormask]


# Best case for 1.05 < z <  1.65 for rioffset, iyoffset, izmin, and gfiblim values
rishift = 0 
iyshift = 0.06636818841724967
izmin = 0.36442580263398095

color_cuts_clean = np.logical_and((ri < iy - 0.19 + rishift),
                             (iy > 0.35 + iyshift))
color_cuts_clean &= iz > izmin


# color cuts using the noisey y band and an ri min. This seems to be as effective as an iy min  
rimin = 0.22
ri_intercept = 0.17
color_cuts_noise = np.logical_and(
    (ri < iy_noise - ri_intercept),
    (ri > rimin))
color_cuts_noise &= iz > izmin

# new best color cuts using the noisey y band. Moving the iy min threshold slightly and an ri < iz cut throwout a good amount of contaminants
iz_noise_cut = 0.35
ri_noise_cut = 0.12
iy_noise_cut = 0.50

color_cuts_noise_new = np.logical_and(
    (ri < iz + ri_noise_cut),
    (iz > iz_noise_cut))
color_cuts_noise_new &= iy_noise > iy_noise_cut
 
clean_comp, clean_pure = complete_purity_cutsonly(color_cuts_clean, photoz, zcut = [1.05,1.55])
print(f'the completeness and purity of the color cuts used for our final optimal selection with no degraded y band are {clean_comp:.03}, {clean_pure:.03}')
noise_comp, noise_pure = complete_purity_cutsonly(color_cuts_noise, photoz, zcut = [1.05,1.55])
print(f'ri < iy_noise - {ri_intercept:.03} & ri > {rimin:.03} & iz > {izmin:.03}: completeness = {noise_comp:.03}, purity = {noise_pure:.03}')
noise_new_comp, noise_new_pure = complete_purity_cutsonly(color_cuts_noise_new, photoz, zcut = [1.05,1.55])
print(f'ri < iz + {ri_noise_cut:.03} & iz > {iz_noise_cut} & iy_noise > {iy_noise_cut}: completeness = {noise_new_comp:.03}, purity = {noise_new_pure:.03}')

#plot the photoz distribution of all three samples above 'z 5$\sigma$ depth'
noise = 'noise'
fig, ax = plt.subplots(figsize=(8,6))
#ax.hist(photoz, bins = 100, alpha = 0.5, label = 'all galaxies')
ax.hist(photoz[color_cuts_clean], bins = np.linspace(0, 2, 100), histtype= 'step', color = 'blue', alpha = 0.5, label = f'clean y optimal cuts')
ax.hist(photoz[color_cuts_noise], bins = np.linspace(0, 2, 100), alpha = 0.5, histtype= 'step', color = 'orange', label = f'r-i < i-y$_{{noise}}$ - {ri_intercept:.2} & r-i > {rimin:.2} & i-z > {izmin:.2}')
ax.hist(photoz[color_cuts_noise_new], bins = np.linspace(0, 2, 100), alpha = 0.5, histtype= 'step', color = 'green', label = f'r-i < i-z + {ri_noise_cut:.2} & i-z > {iz_noise_cut:.2} & i-y$_{{noise}}$ > {iy_noise_cut:.2}')
ax.axvline(x = 1.05, ls = '--', color='black')
ax.axvline(x = 1.55, ls = '--', color='black')
ax.set_xlabel('LePhare Photometric Redshift', fontsize = 16)
ax.set_ylabel('Count', fontsize = 16)
ax.xaxis.set_tick_params(labelsize = 12)
ax.yaxis.set_tick_params(labelsize = 12)
ax.legend(loc = 'upper left', fontsize = 10)
plt.savefig('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/yband_noise_hists.png', dpi=300, bbox_inches='tight')