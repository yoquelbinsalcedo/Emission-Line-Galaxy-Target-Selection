import numpy as np
from matplotlib import pyplot as plt
from astropy.io.misc.hdf5 import read_table_hdf5
import pandas as pd


def plot_and_stats(z_spec_cos, z_phot_cos, z_spec_hsc, z_phot_hsc, ylabel, ylabel_hsc, vlinemin=1.1, vlinemax=1.6, filepath='/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/paper_figures/script_figure_truth/specz_vs_photz_lephare_hsc.png'):

    x = np.arange(0, 5.4, 0.05)

# define differences of >0.15*(1+z) as non-Gaussian 'outliers'
    outlier_upper = x + 0.15*(1+x)
    outlier_lower = x - 0.15*(1+x)

    mask_cos = np.abs((z_phot_cos - z_spec_cos)/(1 + z_spec_cos)) > 0.15
    notmask_cos= ~mask_cos
  
# Standard Deviation of the predicted redshifts compared to the data:
    std_result = np.std((z_phot_cos - z_spec_cos)/(1 + z_spec_cos), ddof=1)

# Normalized MAD (Median Absolute Deviation):
    nmad_cos = 1.48 * np.median(np.abs((z_phot_cos - z_spec_cos)/(1 + z_spec_cos)))

#Percentage of delta-z > 0.15(1+z) outliers:
    eta_cos = np.sum(np.abs((z_phot_cos - z_spec_cos)/(1 + z_spec_cos)) > 0.15)/len(z_spec_cos)
    
    #Median offset (normalized by (1+z); i.e., bias:
    bias_cos = np.median(((z_phot_cos - z_spec_cos)/(1 + z_spec_cos)))

    sigbias_cos =std_result/np.sqrt(0.64*len(z_phot_cos))
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
    ax0.plot(z_spec_cos[mask_cos], z_phot_cos[mask_cos], 'r.', markersize=6,  alpha=0.5)
    ax0.plot(z_spec_cos[notmask_cos], z_phot_cos[notmask_cos], 'b.',  markersize=6, alpha=0.5)
    ax0.plot(x, x, linewidth=1.5, color = 'red')
    ax0.set_title('$\sigma_\mathrm{NMAD} \ = $%6.4f\n'%nmad_cos+ '$f_{\mathrm{outliers}}$ = %6.1f'%(eta_cos*100)+'%', fontsize=22)
    ax0.set_xlim(0, 3)
    ax0.set_ylim(0, 3)
    ax0.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 33)
    ax0.set_ylabel(ylabel, fontsize = 33)
    ax0.xaxis.set_tick_params(labelsize = 20)
    ax0.yaxis.set_tick_params(labelsize = 20)
    
    # ax1 for the histogram
    ax1.set_box_aspect(1)
    ax1.hist(z_spec_cos, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'black', label = 'spec-z')
    ax1.hist(z_phot_cos, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'red', label = 'photo-z')
    ax1.axvline(x = vlinemin,ls= '--', color='black')
    ax1.axvline(x = vlinemax,ls= '--', color='black')
    ax1.set_xlabel('redshift', fontsize = 33)
    ax1.set_ylabel('count', fontsize = 33)
    ax1.legend(loc = 'upper right', fontsize = 20)
    ax1.xaxis.set_tick_params(labelsize = 20)
    ax1.yaxis.set_tick_params(labelsize = 20)
    
    # add lines to indicate outliers
    ax2.set_box_aspect(1)
    ax2.plot(x, outlier_upper, 'k--')
    ax2.plot(x, outlier_lower, 'k--')
    ax2.plot(z_spec_hsc[mask_hsc], z_phot_hsc[mask_hsc], 'r.', markersize=6,  alpha=0.5)
    ax2.plot(z_spec_hsc[notmask_hsc], z_phot_hsc[notmask_hsc], 'b.',  markersize=6, alpha=0.5)
    ax2.plot(x, x, linewidth=1.5, color = 'red')
    ax2.set_title('$\sigma_\mathrm{NMAD} \ = $%6.4f\n'%nmad_hsc+ '$f_{\mathrm{outliers}}$ = %6.1f'%(eta_hsc*100)+'%', fontsize=22)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 3)
    ax2.set_xlabel('$z_{\mathrm{spec}}$', fontsize = 33)
    ax2.set_ylabel(ylabel_hsc, fontsize = 33)
    ax2.xaxis.set_tick_params(labelsize = 20)
    ax2.yaxis.set_tick_params(labelsize = 20)
    
    # ax1 for the histogram
    ax3.set_box_aspect(1)
    ax3.hist(z_spec_hsc, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'black', label = 'spec-z')
    ax3.hist(z_phot_hsc, bins = np.linspace(0, 3, 100), histtype= 'step', color = 'red', label = 'photo-z')
    ax3.axvline(x = vlinemin,ls= '--', color='black')
    ax3.axvline(x = vlinemax,ls= '--', color='black')
    ax3.set_xlabel('redshift', fontsize = 33)
    ax3.set_ylabel('count', fontsize = 33)
    ax3.legend(loc = 'upper right', fontsize = 20)
    ax3.xaxis.set_tick_params(labelsize = 20)
    ax3.yaxis.set_tick_params(labelsize = 20)
    plt.savefig(filepath, dpi = 300, bbox_inches='tight')

    plt.subplots_adjust(wspace=-0.3)
    #plt.show()


# load in crossmatched cat of cosmos2020 and spec truth elgs and the full hsc wide cat

cosmos_crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/cosmos2020_spec_truth_hsc_wide_crossmatch.hdf5'
combined_cat_cos = read_table_hdf5(input=cosmos_crossmatch_file_path)

o2_snr_comb_cos = combined_cat_cos['OII_FLUX']*np.sqrt(combined_cat_cos['OII_FLUX_IVAR'])   
chi2_comb_cos = combined_cat_cos['DELTACHI2']
snr_mask_comb_cos = o2_snr_comb_cos > 10**(0.9 - 0.2*np.log10(chi2_comb_cos))
snr_mask_comb_cos = np.logical_or(snr_mask_comb_cos, chi2_comb_cos > 25)


# # apply our final optimized cuts
# optimization_csv_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/optimization_results/spec_truth/1-1_1-6_opt_params.csv'
# opt_params = pd.read_csv(optimization_csv_path)
# rishift, iyshift, izmin, gfiblim = opt_params['rishift'][0], opt_params['iyshift'][0], opt_params['izmin'][0], opt_params['gfiberlim'][0]

# target_iyri_cut = -0.19
# target_iy_cut = 0.35
# color_mask_iyri_cos = np.logical_and(
#     (combined_cat_cos['r_mag'] - combined_cat_cos['i_mag'] < combined_cat_cos['i_mag'] - combined_cat_cos['y_mag'] + target_iyri_cut + rishift),
#     (combined_cat_cos['i_mag'] - combined_cat_cos['y_mag'] > target_iy_cut + iyshift))
# color_mask_iz_cos = (combined_cat_cos['i_mag'] - combined_cat_cos['z_mag']) > izmin
# gfiber_mask_cos = combined_cat_cos['g_fiber_mag'] < gfiblim
# opt_cuts_cos = np.logical_and.reduce((color_mask_iyri_cos, color_mask_iz_cos, gfiber_mask_cos))

photoz_cos = combined_cat_cos['photoz'][snr_mask_comb_cos]
specz_cos = combined_cat_cos['Z'][snr_mask_comb_cos]


# load in the cross-match cat of hsc with grizy photozs with spec truth elgs

hsc_crossmatch_file_path = '/Users/yokisalcedo/Desktop/Emission-Line-Galaxy-Target-Selection/data/processed/hdf5/spec_truth_hsc_photz_crossmatch.hdf5'
combined_cat_hsc = read_table_hdf5(input=hsc_crossmatch_file_path)

o2_snr_comb_hsc = combined_cat_hsc['OII_FLUX']*np.sqrt(combined_cat_hsc['OII_FLUX_IVAR'])
chi2_comb_hsc = combined_cat_hsc['DELTACHI2']
snr_mask_comb_hsc = o2_snr_comb_hsc > 10**(0.9 - 0.2*np.log10(chi2_comb_hsc))
snr_mask_comb_hsc = np.logical_or(snr_mask_comb_hsc, chi2_comb_hsc > 25)

# color_mask_iyri_hsc = np.logical_and(
#     (combined_cat_hsc['r_mag'] - combined_cat_hsc['i_mag'] < combined_cat_hsc['i_mag'] - combined_cat_hsc['y_mag'] + target_iyri_cut + rishift),
#     (combined_cat_hsc['i_mag'] - combined_cat_hsc['y_mag'] > target_iy_cut + iyshift))
# color_mask_iz_hsc = (combined_cat_hsc['i_mag'] - combined_cat_hsc['z_mag']) > izmin
# gfiber_mask_hsc = combined_cat_hsc['g_fiber_mag'] < gfiblim
# opt_cuts_hsc = np.logical_and.reduce((color_mask_iyri_hsc, color_mask_iz_hsc, gfiber_mask_hsc))

photoz_hsc = combined_cat_hsc['photoz_best'][snr_mask_comb_hsc]
specz_hsc = combined_cat_hsc['Z'][snr_mask_comb_hsc]

plot_and_stats(specz_cos, photoz_cos, specz_hsc, photoz_hsc, ylabel='Lephare photo-z', ylabel_hsc='HSC photo-z' , vlinemin= 1.1, vlinemax = 1.6)
