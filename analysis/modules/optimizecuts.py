import numpy as np

# Below are the functions used to optimize the ELG target selection cuts at high redshift for two different samples 

def get_surf_density_pilot(cat_full, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the surface density'''
    #define the color cuts
    rishift_min = np.minimum(0,rishift)
    cuts_iyri = np.logical_and((cat_full['r_mag'] - cat_full['i_mag'] < cat_full['i_mag'] - cat_full['y_mag'] - 0.19 + rishift_min),
                           (cat_full['i_mag'] - cat_full['y_mag'] > 0.35 + iyshift)) 
    highzcut = (cat_full['i_mag'] - cat_full['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)
    
    #define the magnitude cuts
    gfib_mask = cat_full['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(cat_full['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final = np.logical_and(colorcuts, gbandcut)
    cutcat = cat_full[cuts_final]
    #calculate the surface density
    area = 16
    surf_density = len(cutcat)/area
    return surf_density


def get_success_rate_pilot(catalog, zrange=(1.1, 1.6), ri_targcut=-0.19, iy_targcut=0.35, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the success rate and redshift range success rate'''
    # define the color cuts
    rishift_min = np.minimum(0,rishift)
    cuts_iyri = np.logical_and(
        (catalog['r_mag'] - catalog['i_mag'] < catalog['i_mag'] - catalog['y_mag'] + ri_targcut + rishift_min),
        (catalog['i_mag'] - catalog['y_mag'] > iy_targcut + iyshift)) 
    highzcut = (catalog['i_mag'] - catalog['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)

    # define the magnitude cuts
    gfib_mask = catalog['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(catalog['g_mag'] < glim, gfib_mask)

    # combine the color and magnitude cuts
    # 1400 sec exposure cut
    exposure = catalog['TSNR2_LRG']*12.15
    tmask = exposure < 1400
    cuts_final = np.logical_and(colorcuts, gbandcut)
    
    # quality cuts for reliable speczs
    o2_snr = catalog['OII_FLUX']*np.sqrt(catalog['OII_FLUX_IVAR'])   
    chi2 = catalog['DELTACHI2']
    snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
    snr_o2_mask = np.logical_or(snr_mask, chi2 > 25) 

    # calculate the fraction of objects with good redshifts with the x < t < 1400 sample, where x = 200 sec for pilot and 700 sec for the truth sample
    redshift_success_rate = np.sum(np.logical_and.reduce((cuts_final, tmask, snr_o2_mask)))/np.sum(np.logical_and(cuts_final, tmask))
    
    # calculate the redshift range success rate
    # this should be the ratio of the number of (objs at the correct redshift and with good speczs) to the (total number passing color/mag cuts and snr/o2 quality cut)
    zcut = np.logical_and(catalog['Z'] > zrange[0], catalog['Z'] < zrange[1])
    range_success = np.sum(np.logical_and.reduce((cuts_final, snr_o2_mask, zcut)))/np.sum(np.logical_and(cuts_final, snr_o2_mask))
    
    # calculate the net redshift yield
    redshift_range_yield = range_success*redshift_success_rate
    
    return redshift_success_rate, redshift_range_yield

# optimization wrapper for the pilot survey data


def opt_wrapper_pilot(x, combined_cat, hsc_cat, zrange=(1.1, 1.6), w=0.1):
    deltari, deltaiy, izmin, gfiblim = x
    density = get_surf_density_pilot(hsc_cat, rishift = deltari, iyshift = deltaiy, izmin=izmin, gfiblim=gfiblim)
    zsuccess, rangesuccess = get_success_rate_pilot(combined_cat, zrange=zrange, rishift=deltari, iyshift=deltaiy, izmin=izmin, gfiblim=gfiblim)
    density_yield = density * rangesuccess
    target = 1370
    loss = -rangesuccess*100 + w*(density_yield - target)**2
    return loss


# Below are the functions to be used for the spectroscopic truth sample data

def get_surf_density_truth(cat_full, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the surface density'''
    #define the color cuts
    cuts_iyri = np.logical_and((cat_full['r_mag'] - cat_full['i_mag'] < cat_full['i_mag'] - cat_full['y_mag'] - 0.19 + rishift),
                           (cat_full['i_mag'] - cat_full['y_mag'] > 0.35 + iyshift)) 
    highzcut = (cat_full['i_mag'] - cat_full['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)
    
    #define the magnitude cuts
    gfib_mask = cat_full['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(cat_full['g_mag'] < glim, gfib_mask)
    #combine the color and magnitude cuts
    cuts_final = np.logical_and(colorcuts, gbandcut)
    cutcat = cat_full[cuts_final]
    #calculate the surface density
    area = 16
    surf_density = len(cutcat)/area
    return surf_density


def get_success_rate_truth(catalog, zrange=(1.1, 1.6), ri_targcut=-0.19, iy_targcut=0.35, rishift=0, iyshift=0, izmin=-99, glim=99, gfiblim=99):
    '''This function takes in a catalog and the color cuts to use and outputs the success rate and redshift range success rate'''
    # define the color cuts
    cuts_iyri = np.logical_and(
        (catalog['r_mag'] - catalog['i_mag'] < catalog['i_mag'] - catalog['y_mag'] + ri_targcut + rishift),
        (catalog['i_mag'] - catalog['y_mag'] > iy_targcut + iyshift)) 
    highzcut = (catalog['i_mag'] - catalog['z_mag']) > izmin
    colorcuts = np.logical_and(cuts_iyri, highzcut)

    # define the magnitude cuts
    gfib_mask = catalog['g_fiber_mag'] < gfiblim
    gbandcut = np.logical_and(catalog['g_mag'] < glim, gfib_mask)

    # combine the color and magnitude cuts
    # 1400 sec exposure cut
    exposure = catalog['EFFTIME']
    tmask = exposure < 1400
    cuts_final = np.logical_and(colorcuts, gbandcut)
    
    # quality cuts for reliable speczs
    o2_snr = catalog['OII_FLUX']*np.sqrt(catalog['OII_FLUX_IVAR'])   
    chi2 = catalog['DELTACHI2']
    snr_mask = o2_snr > 10**(0.9 - 0.2*np.log10(chi2))
    snr_o2_mask = np.logical_or(snr_mask, chi2 > 25) 


    # calculate the fraction of objects with good redshifts with the 700 < t < 1400 sample 
    redshift_success_rate = np.sum(np.logical_and.reduce((cuts_final, tmask, snr_o2_mask)))/np.sum(np.logical_and(cuts_final, tmask))
    
    # calculate the redshift range success rate for t > 700 sample
    zcut = np.logical_and(catalog['Z'] > zrange[0], catalog['Z'] < zrange[1])
    range_success = np.sum(np.logical_and.reduce((cuts_final, snr_o2_mask, zcut)))/np.sum(np.logical_and(cuts_final, snr_o2_mask))
    
    # calculate the net redshift yield
    redshift_range_yield = range_success*redshift_success_rate
    
    return redshift_success_rate, redshift_range_yield


def opt_wrapper_truth(x, combined_cat, hsc_cat, zrange=(1.1, 1.6), w=0.1, target=1370):
    deltari, deltaiy, izmin, gfiblim = x
    density = get_surf_density_truth(hsc_cat, rishift=deltari, iyshift=deltaiy, izmin=izmin, gfiblim=gfiblim)
    zsuccess, rangesuccess = get_success_rate_truth(combined_cat, zrange=zrange, rishift=deltari, iyshift=deltaiy, izmin=izmin, gfiblim=gfiblim)
    density_yield = density * rangesuccess
    target = target
    loss = -rangesuccess*100 + w*(density_yield - target)**2 
    return loss
