# Emission-Line-Galaxy-Target-Selection-
 
  This is a repository for the code used for an emission line galaxy (ELG) target selection for DESI-II using deep HSC Wide grizy imaging and photometric redshifts from the COSMOS2020 catalog. The HSC Wide imaging will serve as the best prototype for desgining ELG samples that could be selected with early LSST imaging.

## How to use the scripts in analysis folder:
You will need **python 3.8 or higher**
### What order to run cosmos2020 scripts:
**1.)** train_all_feature_random_forest_1.1_1.6
**2.)** plot_permutation_feature_importance_1.1_1.6
**3.)** train_best_feature_random_forest_1.1_1.6
**4.)** plot_color_color_best_feature_random_forest_prob_1.1_1.6
**5.)** plot_roc_curve_all_feature_1.1_1.6
**6.)** plot_roc_curve_best_feature_1.1_1.6
**7.)** plot_color_color_best_feature_random_forest_prob_selection_1.1_1.6
**8.)** plot_color_cut_selection_1.1_1.6
**9.)** plot_hist_color_cut_selection_1.1_1.6
**10.)** plot_hist_best_selections_1.1_1.6
