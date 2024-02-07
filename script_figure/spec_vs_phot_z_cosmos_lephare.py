#imports
import numpy as np
from astropy import units as u 
from astropy import table
import astropy.units as u
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.table import hstack

def plot_and_stats(z_spec,z_phot, ylabel, isvert = True, vlinemin = 1.05, vlinemax = 1.65, filepath = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure/spec_vs_phot_z_cosmos_lephare.png'):
    
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
    #plt.show()
    

#Load in catalogs 

#load in specz cataloges 
tert = table.Table.read('/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/all_elgs.fits',format='fits',hdu=1)
#cleaning specz catalog
elgmask = tert['TERTIARY_TARGET'] == 'ELG'
fiber_status = tert['COADD_FIBERSTATUS'] == 0 
exposure = tert['TSNR2_LRG']*12.15
tmask = exposure > 200
t_mask = np.logical_and.reduce((tmask, fiber_status, elgmask, tert['YSH'] == True))
elgs = tert[t_mask]

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
cat = cat[colormaskx]

#merge both cosmos2020 and elgs catalogs 
ra_cos = cat['RA']
dec_cos = cat['DEC']
ra_elg = elgs['TARGET_RA']
dec_elg = elgs['TARGET_DEC']
cos_coord = SkyCoord(ra_cos, dec_cos)
elg_coord = SkyCoord(ra_elg*u.degree, dec_elg*u.degree)
idx, d2d, d3d = elg_coord.match_to_catalog_sky(cos_coord)
dmask_cos = d2d.arcsec < 1
combined_cat_cos = hstack([elgs, cat[idx]])[dmask_cos]
print(len(combined_cat_cos))
print(combined_cat_cos['photoz'])

photoz_cos = combined_cat_cos['photoz']
specz_cos = combined_cat_cos['Z']

o2_snr_comb_cos = combined_cat_cos['OII_FLUX']*np.sqrt(combined_cat_cos['OII_FLUX_IVAR'])   
chi2_comb_cos = combined_cat_cos['DELTACHI2']
snr_mask_comb_cos = o2_snr_comb_cos > 10**(0.9 - 0.2*np.log10(chi2_comb_cos))
snr_mask_comb_cos = np.logical_or(snr_mask_comb_cos, chi2_comb_cos > 25)

#photz vs. specz using hsc photoz
plot_and_stats(specz_cos[snr_mask_comb_cos], photoz_cos[snr_mask_comb_cos], ylabel = 'Cosmos2020 Lephare photoz', isvert = True)

