# Emission-Line-Galaxy-Target-Selection-
 
  This is a repository for the code used for an emission line galaxy (ELG) target selection for DESI-II using deep HSC Wide grizy imaging and photometric redshifts from the COSMOS2020 catalog. The HSC wide imaging is the best proxy for desgining ELG samples that could be selected with early Large Synoptic Survey Telescope (LSST) imaging.

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

### Running pilot/truth optimization scripts:
Pilot and Truth refers to two different spectroscopic samples that were obtained by the Dark Energy Spectroscopic Instrument (DESI). The color cuts used to obtain the **Pilot Sample** are the same as those applied in the **plot_color_cut_selection_1.1_1.6** cosmos2020 script. The purpose of this sample was to test a target selection of high redshift (1.1 < z < 1.6) ELGs using HSC wide imaging as a prototype for upcoming LSST imaging. The **Truth Sample** was obtained using broader color cuts than the current DESI ELG sample in order to study how imaging systematics affect DESI ELGs
