# Design of high redshift emission-line galaxy target selections for DESI-II
 
  This is a repository for the code used for an emission-line galaxy (ELG) target selection for DESI-II using deep HSC wide *grizy* imaging and photometric redshifts from the COSMOS2020 catalog. The HSC wide imaging is the best proxy for desgining ELG samples that could be selected with early Large Synoptic Survey Telescope (LSST) imaging.

## How to use the scripts in analysis folder:
**Dependencies**
* numpy
* pandas
* astropy
* matplotlib
* sklearn

### What order to run cosmos2020 scripts:
**1.)** train_all_feature_random_forest_1.1_1.6<br>
**2.)** plot_permutation_feature_importance_1.1_1.6<br>
**3.)** train_best_feature_random_forest_1.1_1.6<br>
**4.)** plot_color_color_best_feature_random_forest_prob_1.1_1.6<br>
**5.)** plot_roc_curve_all_feature_1.1_1.6<br>
**6.)** plot_roc_curve_best_feature_1.1_1.6<br>
**7.)** plot_color_color_best_feature_random_forest_prob_selection_1.1_1.6<br>
**8.)** plot_color_cut_selection_1.1_1.6<br>
**9.)** plot_hist_color_cut_selection_1.1_1.6<br>
**10.)** plot_hist_best_selections_1.1_1.6<br>

### Short description of pilot and truth samples:
Pilot and Truth refers to two different spectroscopic samples that were obtained by the Dark Energy Spectroscopic Instrument (DESI). The color cuts used to obtain the **Pilot Sample** were *r* - *i* < *i* - *y* - 0.19 and *i* - *y* > 0.35. This sample was used test a target selection of high redshift (1.1 < z < 1.6) ELGs using HSC wide imaging as a prototype for upcoming LSST imaging. The **Truth Sample** was obtained using broader color cuts than the current DESI ELG sample in order to study how imaging systematics affect DESI ELGs. This sample was also used to optimize a high redshift ELG selection using simple color cuts.

### optimizecuts.py module:
Make sure this is in a directory that **python knows the path of**. This is a module which contains functions to calculate **target density**, **redshift success rate**, **redshift range success rate**, and **net target density yield** for both **Pilot** and **Truth** samples.<br> These functions require:<br> 
* set of g-fiber limiting magnitudes
* shift to the *r* - *i* < *i* - *y* - 0.19 color cut
* shift to the *i* - *y* > 0.35 color cut
* *i* - *z* minimum
These are applied to the HSC wide catalog and a spectroscopic cross-matched catalog. This also contains wrapper functions to be passed into the `scipy.optimize.opt` routine for optimization where the free parameters are the function arguments above.

### Optimization pilot and truth samples 
