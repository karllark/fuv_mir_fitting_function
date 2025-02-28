import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval

from dust_extinction.averages import GCC09_MWAvg, G03_SMCBar, G24_SMCAvg

from helpers import G25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ave",
        help="average extinction curve to plot",
        default="GCC09_MWAvg",
        choices=["GCC09_MWAvg", "G24_SMCAvg"],
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=3)
    plt.rc("axes", linewidth=3)
    plt.rc("xtick.major", width=3, size=10)
    plt.rc("xtick.minor", width=2, size=5)
    plt.rc("ytick.major", width=3, size=10)
    plt.rc("ytick.minor", width=2, size=5)

    nrows = 2
    hratios = [3, 1]
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(12, 9),
        sharex="col",
        gridspec_kw={
            "width_ratios": [1],
            "height_ratios": hratios,
            "wspace": 0.01,
            "hspace": 0.01,
        },
        constrained_layout=True,
    )

    start_wave = 0.04
    if args.ave == "G03_SMCBar":
        gmod = G03_SMCBar()
        xvals = gmod.obsdata_x
        xvals = (1.0 / xvals) * u.micron
        yvals = gmod.obsdata_axav
        yvals_unc = yvals * 0.01
    if args.ave == "G24_SMCAvg":
        gmod = G24_SMCAvg()
        xvals = gmod.obsdata_x
        xvals = (1.0 / xvals) * u.micron
        yvals = gmod.obsdata_axav
        yvals_unc = gmod.obsdata_axav_unc
    else:
        gmod = GCC09_MWAvg()
        xvals = np.concatenate(
            [gmod.obsdata_x_fuse, gmod.obsdata_x_iue, gmod.obsdata_x_bands]
        )
        xvals = (1.0 / xvals) * u.micron
        yvals = np.concatenate(
            [gmod.obsdata_axav_fuse, gmod.obsdata_axav_iue, gmod.obsdata_axav_bands]
        )
        yvals_unc = np.concatenate(
            [
                gmod.obsdata_axav_unc_fuse,
                gmod.obsdata_axav_unc_iue,
                gmod.obsdata_axav_unc_bands,
            ]
        )

    fmod = G25()

    # remove ISS features, no models have them
    fmod.iss1_amp = 0.0
    fmod.iss1_amp.fixed = True
    fmod.iss2_amp = 0.0
    fmod.iss2_amp.fixed = True
    fmod.iss3_amp = 0.0
    fmod.iss3_amp.fixed = True

    # 20 um silicate well defined
    fmod.sil1_asym.fixed = False
    fmod.sil2_center.fixed = False
    fmod.sil2_fwhm.fixed = False
    fmod.sil2_asym.fixed = False

    ax[0].plot(xvals, yvals, "b-", alpha=0.75)
    ax[0].plot(xvals, fmod(xvals), "k:", alpha=0.5)

    # fit
    fit = LevMarLSQFitter()
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=5.0)

    cmodelfit, mask = or_fit(fmod, 1.0 / xvals.value, yvals, weights=1.0 / yvals_unc)
    ax[0].plot(xvals, cmodelfit(xvals), color="k", alpha=0.5)

    cmodelfit.pprint_parameters()

    amps = ["bkg", "fuv", "bump", "iss1", "iss2", "iss3", "sil1", "sil2", "fir"]
    comps = copy.deepcopy(cmodelfit)
    for camp in amps:
        setattr(comps, f"{camp}_amp", 0.0)
    for camp in amps:
        setattr(comps, f"{camp}_amp", getattr(cmodelfit, f"{camp}_amp"))
        ax[0].plot(xvals, comps(xvals), "k--", alpha=0.5)
        setattr(comps, f"{camp}_amp", 0.0)

    ax[1].plot(xvals, 100.0 * (cmodelfit(xvals) - yvals), "k-", alpha=0.5)
    ax[1].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)

    ax[0].set_yscale("log")
    ax[0].set_ylim(0.0001, 20.0)

    ax[1].set_xscale("log")
    ax[1].set_ylim(-25, 25)

    fname = f"obs_avs_{args.ave}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
