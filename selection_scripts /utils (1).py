import numpy as np
import matplotlib.pyplot as plt
# from desitarget.geomask import imaging_mask

def hist_on_binned_array(hist, edges, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = (edges[1:] + edges[:-1]) / 2
    n, edges, patches = ax.hist(x, weights=hist, bins=edges, **kwargs)
    return n, edges, patches

def fluxToMag(flux):
    return 22.5 - 2.5 * np.log10(flux)


def LRG_quality_cut(sweep):
    """
    Implements basic quality cuts for LRGs
    Args:
    sweep (astropy.table): Legacy survey sweep file

    Returns:
    Boolean array: Selected objects which pass the quality cuts
    astropy.table: sweep file with only the selected rows
    """
    # The Unique object `BRICK_PRIMARY = "T"` cut has already been applied for sweep files
    # Quality in r
    mask = (sweep["FLUX_IVAR_R"] > 0) & (sweep["FLUX_R"] > 0)
    # Quality in z
    mask &= (
        (sweep["FLUX_IVAR_Z"] > 0) & (sweep["FLUX_Z"] > 0) & (sweep["FIBERFLUX_Z"] > 0)
    )
    # Quality in W1
    mask &= (sweep["FLUX_IVAR_W1"] > 0) & (sweep["FLUX_W1"] > 0)
    # Observed in every band
    mask &= (sweep["NOBS_G"] > 0) & (sweep["NOBS_R"] > 0) & (sweep["NOBS_Z"] > 0)
    
    mask &= (sweep["GAIA_PHOT_G_MEAN_MAG"] == 0) | (sweep["GAIA_PHOT_G_MEAN_MAG"] > 18)  # remove bright GAIA sources

    # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
    mask &= sweep["FIBERTOTFLUX_G"] < 10**(-0.4*(17.5-22.5))
    
    # mask &= imaging_mask(sweep["MASKBITS"])
    # BRIGHT, GALAXY, CLUSTER masking
    mask &= (
        ((sweep["MASKBITS"] & 2 ** 1) == 0)
        & ((sweep["MASKBITS"] & 2 ** 12) == 0)
        & ((sweep["MASKBITS"] & 2 ** 13) == 0)
    )

    return mask, sweep[mask]


def LRG_SV_cut(sweep, selection="OPT", survey="south"):
    """Implements SV cuts for LRGs
    Args:
    sweep (astropy.table): Legacy survey sweep file with good quality entries
    selection (str): `OPT` or `IR` which selection cut to apply
    survey (str): `north` or `south`, which survey patch to select

    Returns:
    Boolean array: Selected objects which pass the quality cuts
    selected_sweep: sweep file with only the selected rows
    """
    # Get de-reddened magnitudes
    # Calculate the magnitudes
    names = sweep.columns
    if "gmag" not in names:
        sweep["gmag"] = fluxToMag(sweep["FLUX_G"] / sweep["MW_TRANSMISSION_G"])
    if "rmag" not in names:
        sweep["rmag"] = fluxToMag(sweep["FLUX_R"] / sweep["MW_TRANSMISSION_R"])
    if "zmag" not in names:
        sweep["zmag"] = fluxToMag(sweep["FLUX_Z"] / sweep["MW_TRANSMISSION_Z"])
    if "w1mag" not in names:
        sweep["w1mag"] = fluxToMag(sweep["FLUX_W1"] / sweep["MW_TRANSMISSION_W1"])
    if "w2mag" not in names:
        sweep["w2mag"] = fluxToMag(sweep["FLUX_W2"] / sweep["MW_TRANSMISSION_W2"])
    if "gfibermag" not in names:
        sweep["gfibermag"] = fluxToMag(
            sweep["FIBERFLUX_G"] / sweep["MW_TRANSMISSION_G"]
        )
    if "rfibermag" not in names:
        sweep["rfibermag"] = fluxToMag(
            sweep["FIBERFLUX_R"] / sweep["MW_TRANSMISSION_R"]
        )
    if "zfibermag" not in names:
        sweep["zfibermag"] = fluxToMag(
            sweep["FIBERFLUX_Z"] / sweep["MW_TRANSMISSION_Z"]
        )

    if selection == "OPT":

        if survey == "north":
            # LRG_SV_OPT: SV optical selection
            # non-stellar cut
            lrg_sv_opt = (
                sweep["zmag"] - sweep["w1mag"]
                > 0.8 * (sweep["rmag"] - sweep["zmag"]) - 0.8
            )
            # faint limit
            lrg_sv_opt &= (sweep["zmag"] < 21.0) | (sweep["zfibermag"] < 22.0)
            # low-z cut
            mask_red = (sweep["gmag"] - sweep["w1mag"] > 2.57) & (
                sweep["gmag"] - sweep["rmag"] > 1.35
            )
            # ignore low-z cut for faint objects
            mask_red |= (sweep["rmag"] - sweep["w1mag"]) > 1.75
            lrg_sv_opt &= mask_red
            # straight cut for low-z:
            lrg_mask_lowz = sweep["zmag"] < 20.2
            lrg_mask_lowz &= (
                sweep["rmag"] - sweep["zmag"] > (sweep["zmag"] - 17.17) * 0.45
            )
            lrg_mask_lowz &= (
                sweep["rmag"] - sweep["zmag"] > (sweep["zmag"] - 14.14) * 0.19
            )
            # curved sliding cut for high-z:
            lrg_mask_highz = sweep["zmag"] >= 20.2
            lrg_mask_highz &= ((sweep["zmag"] - 23.15) / 1.3) ** 2 + (
                sweep["rmag"] - sweep["zmag"] + 2.5
            ) ** 2 > 4.48 ** 2
            lrg_sv_opt &= lrg_mask_lowz | lrg_mask_highz

            return lrg_sv_opt, sweep[lrg_sv_opt]

        if survey == "south":
            # LRG_SV_OPT: SV optical selection
            # non-stellar cut
            lrg_sv_opt = (
                sweep["zmag"] - sweep["w1mag"]
                > 0.8 * (sweep["rmag"] - sweep["zmag"]) - 0.8
            )
            # faint limit
            lrg_sv_opt &= (sweep["zmag"] < 21.0) | (sweep["zfibermag"] < 22.0)
            # low-z cut
            mask_red = (sweep["gmag"] - sweep["w1mag"] > 2.5) & (
                sweep["gmag"] - sweep["rmag"] > 1.3
            )
            # ignore low-z cut for faint objects
            mask_red |= (sweep["rmag"] - sweep["w1mag"]) > 1.7
            lrg_sv_opt &= mask_red
            # straight cut for low-z:
            lrg_mask_lowz = sweep["zmag"] < 20.2
            lrg_mask_lowz &= (
                sweep["rmag"] - sweep["zmag"] > (sweep["zmag"] - 17.20) * 0.45
            )
            lrg_mask_lowz &= (
                sweep["rmag"] - sweep["zmag"] > (sweep["zmag"] - 14.17) * 0.19
            )
            # curved sliding cut for high-z:
            lrg_mask_highz = sweep["zmag"] >= 20.2
            lrg_mask_highz &= ((sweep["zmag"] - 23.18) / 1.3) ** 2 + (
                sweep["rmag"] - sweep["zmag"] + 2.5
            ) ** 2 > 4.48 ** 2
            lrg_sv_opt &= lrg_mask_lowz | lrg_mask_highz

            return lrg_sv_opt, sweep[lrg_sv_opt]

        else:
            raise ValueError("Not a valid survey type")

    if selection == "IR":

        if survey == "north":
            # LRG_SV_IR: SV IR selection
            # non-stellar cut
            lrg_sv_ir = (
                sweep["zmag"] - sweep["w1mag"]
                > 0.8 * (sweep["rmag"] - sweep["zmag"]) - 0.8
            )
            # faint limit
            lrg_sv_ir &= (sweep["zmag"] < 21.0) | (sweep["zfibermag"] < 22.0)
            # low-z cut
            lrg_sv_ir &= sweep["rmag"] - sweep["w1mag"] > 1.03
            # sliding IR cut
            lrg_mask_slide = (
                sweep["rmag"] - sweep["w1mag"] > (sweep["w1mag"] - 17.44) * 1.8
            )
            # add high-z objects
            lrg_mask_slide |= sweep["rmag"] - sweep["w1mag"] > 3.1
            lrg_sv_ir &= lrg_mask_slide

            return lrg_sv_ir, sweep[lrg_sv_ir]

        if survey == "south":
            # LRG_SV_IR: SV IR selection
            # non-stellar cut
            lrg_sv_ir = (
                sweep["zmag"] - sweep["w1mag"]
                > 0.8 * (sweep["rmag"] - sweep["zmag"]) - 0.8
            )
            # faint limit
            lrg_sv_ir &= (sweep["zmag"] < 21.0) | (sweep["zfibermag"] < 22.0)
            # low-z cut
            lrg_sv_ir &= sweep["rmag"] - sweep["w1mag"] > 1.0
            # sliding IR cut
            lrg_mask_slide = (
                sweep["rmag"] - sweep["w1mag"] > (sweep["w1mag"] - 17.48) * 1.8
            )
            # add high-z objects
            lrg_mask_slide |= sweep["rmag"] - sweep["w1mag"] > 3.1
            lrg_sv_ir &= lrg_mask_slide

            return lrg_sv_ir, sweep[lrg_sv_ir]

        else:
            raise ValueError("Not a valid survey type")
    else:
        raise ValueError("Not a valid selection type")


def LRG_cut(sweep, survey="south"):
    """Implements main survey cuts for LRGs.
      
    Args:
    sweep (astropy.table): Legacy survey sweep file with good quality entries
    survey (str): `north` or `south`, which survey patch to select

    Returns:
    Boolean array: Selected objects which pass the quality cuts
    selected_sweep: sweep file with only the selected rows
    """
    # Get de-reddened magnitudes
    # Calculate the magnitudes
    names = sweep.columns
    if "gmag" not in names:
        sweep["gmag"] = fluxToMag(sweep["FLUX_G"] / sweep["MW_TRANSMISSION_G"])
    if "rmag" not in names:
        sweep["rmag"] = fluxToMag(sweep["FLUX_R"] / sweep["MW_TRANSMISSION_R"])
    if "zmag" not in names:
        sweep["zmag"] = fluxToMag(sweep["FLUX_Z"] / sweep["MW_TRANSMISSION_Z"])
    if "w1mag" not in names:
        sweep["w1mag"] = fluxToMag(sweep["FLUX_W1"] / sweep["MW_TRANSMISSION_W1"])
    if "w2mag" not in names:
        sweep["w2mag"] = fluxToMag(sweep["FLUX_W2"] / sweep["MW_TRANSMISSION_W2"])
    if "gfibermag" not in names:
        sweep["gfibermag"] = fluxToMag(
            sweep["FIBERFLUX_G"] / sweep["MW_TRANSMISSION_G"]
        )
    if "rfibermag" not in names:
        sweep["rfibermag"] = fluxToMag(
            sweep["FIBERFLUX_R"] / sweep["MW_TRANSMISSION_R"]
        )
    if "zfibermag" not in names:
        sweep["zfibermag"] = fluxToMag(
            sweep["FIBERFLUX_Z"] / sweep["MW_TRANSMISSION_Z"]
        )

    if survey == "north":
        # LRG_IR: baseline IR selection
        # non-stellar cut
        lrg_ir = (
            sweep["zmag"] - sweep["w1mag"]
            > 0.8 * (sweep["rmag"] - sweep["zmag"]) - 0.6
        )
        # faint limit
        lrg_ir &= sweep["zfibermag"] < 21.61
        # low-z cuts
        lrg_ir &= (sweep["gmag"] - sweep["w1mag"] > 2.97) | (sweep["rmag"] - sweep["w1mag"] > 1.8)  
        # double sliding cuts and high-z extension
        lrg_ir &= (
            ((sweep["rmag"] - sweep["w1mag"] > (sweep["w1mag"] - 17.13) * 1.83)
             & (sweep["rmag"] - sweep["w1mag"] > (sweep["w1mag"] - 16.31) * 1.))
            | (sweep["rmag"] - sweep["w1mag"] > 3.4)
        )  

        return lrg_ir, sweep[lrg_ir]

    if survey == "south":
        # LRG_IR: baseline IR selection
        # non-stellar cut
        lrg_ir = (
            sweep["zmag"] - sweep["w1mag"]
            > 0.8 * (sweep["rmag"] - sweep["zmag"]) - 0.6
        )
        # faint limit
        lrg_ir &= sweep["zfibermag"] < 21.6
        # low-z cuts
        lrg_ir &= (sweep["gmag"] - sweep["w1mag"] > 2.9) | (sweep["rmag"] - sweep["w1mag"] > 1.8)  
        # double sliding cuts and high-z extension
        lrg_ir &= (
            ((sweep["rmag"] - sweep["w1mag"] > (sweep["w1mag"] - 17.14) * 1.8)
             & (sweep["rmag"] - sweep["w1mag"] > (sweep["w1mag"] - 16.33) * 1.))
            | (sweep["rmag"] - sweep["w1mag"] > 3.3)
        )  

        return lrg_ir, sweep[lrg_ir]

    else:
        raise ValueError("Not a valid survey type")

        
def ts_plot(cat=None, extra_cat = None, cat_frac = 0.001, extra_cat_frac = 0.002):
    if cat is not None:
        subsample = np.random.choice(len(cat), size= int(cat_frac*len(cat)))
        plot_cat = cat[subsample]
    if extra_cat is not None:
        subsample = np.random.choice(len(extra_cat), size= int(extra_cat_frac*len(extra_cat)))
        plot_extra_cat = extra_cat[subsample]
    fig, axs = plt.subplots(3,2, figsize=(10, 14))
    axs = axs.ravel()
    plt.delaxes(axs[-1])
    #r-z vs z-w1
    if cat is not None:
        axs[0].scatter( (plot_cat["rmag"]-plot_cat["zmag"]), (plot_cat["zmag"]-plot_cat["w1mag"]),
                       s=0.1, c = plot_cat["zphot"], cmap="Dark2_r", vmin=0, vmax=1.2 )
    if extra_cat is not None:
        axs[0].scatter( (plot_extra_cat["rmag"]-plot_extra_cat["zmag"]),
                       (plot_extra_cat["zmag"]-plot_extra_cat["w1mag"]), s=0.1, c = "k")
    x = np.linspace(0, 3,20)
    y = 0.8*x-0.6
#     axs[0].plot(x,y, ls="--", c="gray")
    axs[0].set_xlabel("r-z", fontsize=20)
    axs[0].set_ylabel("z-W1", fontsize=20)
    axs[0].set_xlim(0,3)
    axs[0].set_ylim(-1,3)
    
    #r-w1 vs g-r
    if cat is not None:
        axs[1].scatter( (plot_cat["rmag"]-plot_cat["w1mag"]), (plot_cat["gmag"]-plot_cat["rmag"]),
                   s=0.1, c = plot_cat["zphot"], cmap="Dark2_r", vmin=0, vmax=1.2 )
    if extra_cat is not None:
        axs[1].scatter( (plot_extra_cat["rmag"]-plot_extra_cat["w1mag"]), (plot_extra_cat["gmag"]-plot_extra_cat["rmag"]),
                   s=0.1, c = "k")
    x1 = np.linspace(-1, 1.2,5)
    y1 = -x1+2.6
#     axs[1].plot(x1,y1, ls="--", c="gray")
    x2 = np.linspace(1.2, 1.8,5)
    y2 = 0*x2+1.4
#     axs[1].plot(x2,y2, ls="--", c="gray")
    
    y3 = np.linspace(0, 1.4,5)
    x3 = 0*y3 + 1.8 
#     axs[1].plot(x3,y3, ls="--", c="gray")
    
    axs[1].set_xlabel("r-w1", fontsize=20)
    axs[1].set_ylabel("g-r", fontsize=20)
    axs[1].set_xlim(-0.1,4.5)
    axs[1].set_ylim(0,2.5)
    
    #z vs r-z
    if cat is not None:
        axs[2].scatter( plot_cat["zmag"], (plot_cat["rmag"]-plot_cat["zmag"]),
                       s=0.1, c = plot_cat["zphot"], cmap="Dark2_r", vmin=0, vmax=1.2 )
    if extra_cat is not None:
        axs[2].scatter( plot_extra_cat["zmag"], (plot_extra_cat["rmag"]-plot_extra_cat["zmag"]),
                       s=0.1, c = "k" )
#     x = np.linspace(0, 3,20)
#     y = 0.8*x-0.6
#     axs[0].plot(x,y, ls="--", c="k")
    axs[2].set_xlabel("z", fontsize=20)
    axs[2].set_ylabel("r-z", fontsize=20)
    axs[2].set_xlim(16,21)
    axs[2].set_ylim(0.5,2.5)
    
    #w1 vs r-w1
    if cat is not None:
        axs[3].scatter( plot_cat["w1mag"], (plot_cat["rmag"]-plot_cat["w1mag"]),
                       s=0.1, c = plot_cat["zphot"], cmap="Dark2_r", vmin=0, vmax=1.2 )
    if extra_cat is not None:
        axs[3].scatter( plot_extra_cat["w1mag"], (plot_extra_cat["rmag"]-plot_extra_cat["w1mag"]),
                       s=0.1, c = "k",)
#     x = np.linspace(0, 3,20)
#     y = 0.8*x-0.6
#     axs[0].plot(x,y, ls="--", c="k")
    axs[3].set_xlabel("w1", fontsize=20)
    axs[3].set_ylabel("r-w1", fontsize=20)
    axs[3].set_xlim(15,20)
    axs[3].set_ylim(-0.1,4.5)
    
    #w1 vs r-w1
    if cat is not None:
        cbar = axs[4].scatter( plot_cat["zmag"], plot_cat["zfibermag"],
                       s=0.1, c = plot_cat["zphot"], cmap="Dark2_r", vmin=0, vmax=1.2 )
    if extra_cat is not None:
        axs[4].scatter( plot_extra_cat["zmag"], plot_extra_cat["zfibermag"],
                       s=0.1, c = "k",)

    axs[4].axhline(21.5, ls="--", c="gray")
    axs[4].set_xlabel("z", fontsize=20)
    axs[4].set_ylabel("z fiber", fontsize=20)
    axs[4].set_xlim(16,22)
    axs[4].set_ylim(18,24)
    fig.colorbar(cbar)