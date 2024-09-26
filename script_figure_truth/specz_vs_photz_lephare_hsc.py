#imports
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from pathlib import Path
from astropy.table import hstack
from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.coordinates import SkyCoord


def plot_and_stats(z_spec, z_phot, z_spec_hsc, z_phot_hsc, ylabel, ylabel_hsc, vlinemin=1.1, vlinemax=1.6, filepath='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/script_figure_truth/specz_vs_photz_lephare_hsc.png'):

    x = np.arange(0, 5.4, 0.05)

# define differences of >0.15*(1+z) as non-Gaussian 'outliers'
    outlier_upper = x + 0.15*(1+x)
    outlier_lower = x - 0.15*(1+x)

    mask = np.abs((z_phot - z_spec)/(1 + z_spec)) > 0.15
    notmask = ~mask
  
# Standard Deviation of the predicted redshifts compared to the data:
    std_result = np.std((z_phot - z_spec)/(1 + z_spec), ddof=1)

# Normalized MAD (Median Absolute Deviation):
    nmad = 1.48 * np.median(np.abs((z_phot - z_spec)/(1 + z_spec)))

#Percentage of delta-z > 0.15(1+z) outliers:
    eta = np.sum(np.abs((z_phot - z_spec)/(1 + z_spec)) > 0.15)/len(z_spec)
    
    #Median offset (normalized by (1+z); i.e., bias:
    bias = np.median(((z_phot - z_spec)/(1 + z_spec)))

    sigbias=std_result/np.sqrt(0.64*len(z_phot))
    # make photo-z/spec-z plots

    #Do the same calculations on the hsc photozs   
    outlier_upper = x + 0.15*(1+x)
    outlier_lower = x - 0.15*(1+x)

    mask_hsc = np.abs((z_phot_hsc - z_spec_hsc)/(1 + z_spec_hsc)) > 0.15
    notmask_hsc = ~mask_hsc 
    
# Standard Deviation of the predicted redshifts compared to the data:
    std_result_hsc = np.std((z_phot_hsc - z_spec_hsc)/(1 + z_spec_hsc), ddof=1)

# Normalized MAD (Median Absolute Deviation):
    nmad_hsc = 1.48 * np.median(np.abs((z_phot_hsc - z_spec_hsc)/(1 + z_spec_hsc)))

# Percentage of delta-z > 0.15(1+z) outliers:
    eta_hsc= np.sum(np.abs((z_phot_hsc - z_spec_hsc)/(1 + z_spec_hsc)) > 0.15)/len(z_spec_hsc)
    
    #Median offset (normalized by (1+z); i.e., bias:
    bias_hsc = np.median(((z_phot_hsc - z_spec_hsc)/(1 + z_spec_hsc)))

    sigbias_hsc = std_result_hsc/np.sqrt(0.64*len(z_phot_hsc))
    # make photo-z/spec-z plots

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize = (16, 12), constrained_layout = True)
                                                       
    # add lines to indicate outliers
    ax0.set_box_aspect(1)
    ax0.plot(x, outlier_upper, 'k--')
    ax0.plot(x, outlier_lower, 'k--')
    ax0.plot(z_spec[mask], z_phot[mask], 'r.', markersize=6,  alpha=0.5)
    ax0.plot(z_spec[notmask], z_phot[notmask], 'b.',  markersize=6, alpha=0.5)
    ax0.plot(x, x, linewidth=1.5, color = 'red')
    ax0.set_title('$\sigma_\mathrm{NMAD} \ = $%6.4f\n'%nmad+'$(\Delta z)>0.15(1+z) $ outliers = %6.3f'%(eta*100)+'%', fontsize=17)
    ax0.set_xlim(0, 3)
    ax0.set_ylim(0, 3)
    ax0.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 25)
    ax0.set_ylabel(ylabel, fontsize = 25)
    ax0.xaxis.set_tick_params(labelsize = 16)
    ax0.yaxis.set_tick_params(labelsize = 16)
    
    # ax1 for the histogram
    ax1.set_box_aspect(1)
    ax1.hist(z_spec, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'black', label = 'specz')
    ax1.hist(z_phot, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'red', label = 'photoz')
    ax1.axvline(x = vlinemin,ls= '--', color='black')
    ax1.axvline(x = vlinemax,ls= '--', color='black')
    ax1.set_xlabel('Redshift', fontsize = 25)
    ax1.set_ylabel('Count', fontsize = 25)
    ax1.legend(loc = 'upper left', fontsize = 16)
    ax1.xaxis.set_tick_params(labelsize = 16)
    ax1.yaxis.set_tick_params(labelsize = 16)
    
    # add lines to indicate outliers
    ax2.set_box_aspect(1)
    ax2.plot(x, outlier_upper, 'k--')
    ax2.plot(x, outlier_lower, 'k--')
    ax2.plot(z_spec_hsc[mask_hsc], z_phot_hsc[mask_hsc], 'r.', markersize=6,  alpha=0.5)
    ax2.plot(z_spec_hsc[notmask_hsc], z_phot_hsc[notmask_hsc], 'b.',  markersize=6, alpha=0.5)
    ax2.plot(x, x, linewidth=1.5, color = 'red')
    ax2.set_title('$\sigma_\mathrm{NMAD} \ = $%6.4f\n'%nmad_hsc+'$(\Delta z)>0.15(1+z) $ outliers = %6.3f'%(eta_hsc*100)+'%', fontsize=17)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)
    ax2.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 25)
    ax2.set_ylabel(ylabel_hsc, fontsize = 25)
    ax2.xaxis.set_tick_params(labelsize = 16)
    ax2.yaxis.set_tick_params(labelsize = 16)
    
    # ax1 for the histogram
    ax3.set_box_aspect(1)
    ax3.hist(z_spec_hsc, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'black', label = 'specz')
    ax3.hist(z_phot_hsc, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'red', label = 'photoz')
    ax3.axvline(x = vlinemin,ls= '--', color='black')
    ax3.axvline(x = vlinemax,ls= '--', color='black')
    ax3.set_xlabel('Redshift', fontsize = 25)
    ax3.set_ylabel('Count', fontsize = 25)
    ax3.legend(loc = 'upper left', fontsize = 17)
    ax3.xaxis.set_tick_params(labelsize = 16)
    ax3.yaxis.set_tick_params(labelsize = 16)
    plt.savefig(filepath, dpi = 300, bbox_inches='tight')
    #plt.show()


# load in crossmatched cat of cosmos2020 and spec truth elgs and the full hsc wide cat

cosmos_crossmatch_input_name = 'cosmos2020_spec_truth_hsc_wide_crossmatch'
cosmos_crossmatch_input = f'{cosmos_crossmatch_input_name}.hdf5'
cosmos_crossmatch_file_path = str(Path(f'../data/processed/hdf5/{cosmos_crossmatch_input}'))
combined_cat_cos = read_table_hdf5(input=cosmos_crossmatch_file_path)

o2_snr_comb_cos = combined_cat_cos['OII_FLUX']*np.sqrt(combined_cat_cos['OII_FLUX_IVAR'])   
chi2_comb_cos = combined_cat_cos['DELTACHI2']
snr_mask_comb_cos = o2_snr_comb_cos > 10**(0.9 - 0.2*np.log10(chi2_comb_cos))
snr_mask_comb_cos = np.logical_or(snr_mask_comb_cos, chi2_comb_cos > 25)

photoz_cos = combined_cat_cos['photoz'][snr_mask_comb_cos]
specz_cos = combined_cat_cos['Z'][snr_mask_comb_cos]


# load in the cross-match cat of hsc with grizy photozs with spec truth elgs

hsc_crossmatch_input_name = 'spec_truth_hsc_photz_crossmatch'
hsc_crossmatch_input = f'{hsc_crossmatch_input_name}.hdf5'
hsc_crossmatch_file_path = str(Path(f'../data/processed/hdf5/{hsc_crossmatch_input}'))
combined_cat_hsc = read_table_hdf5(input=hsc_crossmatch_file_path)

o2_snr_comb_hsc = combined_cat_hsc['OII_FLUX']*np.sqrt(combined_cat_hsc['OII_FLUX_IVAR'])
chi2_comb_hsc = combined_cat_hsc['DELTACHI2']
snr_mask_comb_hsc = o2_snr_comb_hsc > 10**(0.9 - 0.2*np.log10(chi2_comb_hsc))
snr_mask_comb_hsc = np.logical_or(snr_mask_comb_hsc, chi2_comb_hsc > 25)

photoz_hsc = combined_cat_hsc['photoz_best'][snr_mask_comb_hsc]
specz_hsc = combined_cat_hsc['Z'][snr_mask_comb_hsc]

plot_and_stats(specz_cos, photoz_cos, specz_hsc, photoz_hsc, ylabel='Cosmos2020 Lephare photoz', ylabel_hsc='HSC photoz' , vlinemin= 1.05, vlinemax = 1.65)
