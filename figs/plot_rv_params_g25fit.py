import copy
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import numpy as np

import warnings

from astropy.table import QTable
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.modeling.fitting import LevMarLSQFitter, FittingWithOutlierRemoval


from helpers import G25


def plot_irv_ssamp(
    ax, itab, label, color="k", linestyle="solid", simpfit=False, inst=None, show_rv=False,
):

    # remove bad regions
    bregions = (
        np.array(
            [
                [1190.0, 1235.0],
                [1370.0, 1408.0],
                [1515.0, 1563.0],
                [6560.0, 6570.0],
                [41000.0, 50000.0],
            ]
        )
        * u.AA
    )
    for cbad in bregions:
        bvals = np.logical_and(itab["waves"] > cbad[0], itab["waves"] < cbad[1])
        itab["npts"][bvals] = 0

    # find regions that have large uncertainties in the intercept and remove
    # gindxs = np.where(itab["npts"] > 0)
    # bvals = (itab["hfintercepts"][gindxs] / itab["hfintercepts_std"][gindxs]) < 10
    # itab["npts"][gindxs][bvals] = 0
    # print(itab["waves"][gindxs][bvals])

    # trim ends
    if inst == "IUE":
        bvals = itab["waves"] > 0.3 * u.micron
        itab["npts"][bvals] = 0
    elif inst == "STIS":
        bvals = itab["waves"] > 0.95 * u.micron
        itab["npts"][bvals] = 0
    elif inst == "SpeXLXD":
        bvals = itab["waves"] > 5.5 * u.micron
        itab["npts"][bvals] = 0

    # set to NAN so they are not plotted
    bvals = itab["npts"] == 0
    itab["slopes"][bvals] = np.nan
    itab["intercepts"][bvals] = np.nan
    gvals = itab["npts"] >= 0
    if simpfit:
        for k, cname in enumerate(["intercepts", "slopes", "rmss"]):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle="dashed",
                color="red",
                alpha=0.75,
            )

    if "lmslopes" in itab.colnames:
        itab["lmslopes"][bvals] = np.nan
        itab["lmintercepts"][bvals] = np.nan

        if "lmslopes_std" not in itab.colnames:
            itab["lmslopes_std"] = itab["lmslopes"] * 0.0
            itab["lmintercepts_std"] = itab["lmintercepts"] * 0.0
        for k, cname in enumerate(["lmintercepts", "lmslopes"]):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle="dotted",
                color=color,
                label=label,
                alpha=0.75,
            )

            ax[k * 2].fill_between(
                itab["waves"][gvals].value,
                itab[cname][gvals] - itab[f"{cname}_std"],
                itab[cname][gvals] + itab[f"{cname}_std"],
                color=color,
                alpha=0.25,
            )

        return (
            itab["npts"],
            itab["waves"],
            itab["lmintercepts"],
            itab["lmslopes"],
            itab["lmintercepts_std"],
            itab["lmslopes_std"],
        )

    if "d2slopes" in itab.colnames:
        itab["d2slopes"][bvals] = np.nan
        itab["d2intercepts"][bvals] = np.nan

        if "d2slopes_std" not in itab.colnames:
            itab["d2slopes_std"] = itab["d2slopes"] * 0.0
            itab["d2intercepts_std"] = itab["d2intercepts"] * 0.0
        if show_rv:
            ctypes = ["d2intercepts", "d2slopes"]
        else:
            ctypes = ["d2intercepts"]
        for k, cname in enumerate(ctypes):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle="dotted",
                color=color,
                label=label,
                alpha=0.75,
            )

            ax[k * 2].fill_between(
                itab["waves"][gvals].value,
                itab[cname][gvals] - itab[f"{cname}_std"],
                itab[cname][gvals] + itab[f"{cname}_std"],
                color=color,
                alpha=0.25,
            )

        # cubic fits
        # itab["d2curves_quad"][bvals] = np.nan
        # itab["d2slopes_quad"][bvals] = np.nan
        # itab["d2intercepts_quad"][bvals] = np.nan
        # itab["d2curves_quad_std"] = itab["d2curves_quad"] * 0.1
        # itab["d2slopes_quad_std"] = itab["d2slopes_quad"] * 0.1
        # itab["d2intercepts_quad_std"] = itab["d2intercepts_quad"] * 0.1
        # for k, cname in enumerate(["d2intercepts_quad", "d2slopes_quad", "d2curves_quad"]):
        #     ax[k * 2].plot(
        #         itab["waves"][gvals],
        #         itab[cname][gvals],
        #         linestyle="solid",
        #         color=color,
        #         label=label,
        #         alpha=0.75,
        #     )
        #     ax[k * 2].fill_between(
        #         itab["waves"][gvals].value,
        #         itab[cname][gvals] - itab[f"{cname}_std"],
        #         itab[cname][gvals] + itab[f"{cname}_std"],
        #         color=color,
        #         alpha=0.25,
        #     )

        # likelihood ratios
        # itab["d2lnlikes"][bvals] = np.nan
        # itab["d2lnlikes_quad"][bvals] = np.nan
        # lnratio = itab["d2lnlikes_quad"][gvals] - itab["d2lnlikes"][gvals]
        # ax[4].plot(
        #     itab["waves"][gvals],
        #     lnratio,
        #     linestyle="solid",
        #     color="black",
        #     label=label,
        #     alpha=0.75,
        # )

        return (
            itab["npts"],
            itab["waves"],
            itab["d2intercepts"],
            itab["d2slopes"],
            itab["d2intercepts_std"],
            itab["d2slopes_std"],
        )

    if "mcslopes" in itab.colnames:
        itab["mcslopes"][bvals] = np.nan
        itab["mcintercepts"][bvals] = np.nan
        for k, cname in enumerate(["mcintercepts", "mcslopes"]):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle="dotted",
                color=color,
                label=label,
                alpha=0.75,
            )
            ax[k * 2].fill_between(
                itab["waves"][gvals].value,
                itab[cname][gvals] - itab[f"{cname}_std"],
                itab[cname][gvals] + itab[f"{cname}_std"],
                color=color,
                alpha=0.25,
            )
    if "hfslopes" in itab.colnames:
        itab["hfslopes"][bvals] = np.nan
        itab["hfintercepts"][bvals] = np.nan
        itab["hfsigmas"][bvals] = np.nan
        itab["hfrmss"][bvals] = np.nan
        # for k, cname in enumerate(["hfintercepts", "hfslopes", "hfsigmas"]):
        for k, cname in enumerate(["hfintercepts", "hfslopes"]):
            ax[k * 2].plot(
                itab["waves"][gvals],
                itab[cname][gvals],
                linestyle=linestyle,
                color="black",
                label=label,
                alpha=0.75,
            )
        if "hfslopes_std" in itab.colnames:
            # for k, cname in enumerate(["hfintercepts", "hfslopes", "hfsigmas"]):
            for k, cname in enumerate(["hfintercepts", "hfslopes"]):
                ax[k * 2].fill_between(
                    itab["waves"][gvals].value,
                    itab[cname][gvals] - itab[f"{cname}_std"],
                    itab[cname][gvals] + itab[f"{cname}_std"],
                    color="black",
                    alpha=0.25,
                )


def plot_resid(ax, data, dindx, model, color):
    """
    Plot the residuals to the model
    """
    bvals = data[0] <= 0
    data[dindx][bvals] = np.nan

    # only plot where the model is valid
    gvals = (data[1].value >= 1.0 / model.x_range[1]) & (
        data[1].value <= 1.0 / model.x_range[0]
    )
    fitx = 1.0 / data[1][gvals].value
    ax.plot(
        data[1][gvals],
        (data[dindx][gvals] - model(fitx)) /data[dindx + 2][gvals],
        # data[dindx][gvals] - model(fitx),
        linestyle="dotted",
        color=color,
        alpha=0.75,
    )
    ax.fill_between(
        data[1][gvals].value,
        data[dindx][gvals].value - model(fitx) - data[dindx + 2][gvals].value,
        data[dindx][gvals].value - model(fitx) + data[dindx + 2][gvals].value,
        color=color,
        alpha=0.25,
    )


def plot_wavereg(ax, models, datasets, colors, wrange, no_weights=False, show_rv=False):
    """
    Do the fits and plot the fits and residuals
    """
    warnings.filterwarnings("ignore")

    npts = []
    waves = []
    intercepts = []
    intercepts_unc = []
    slopes = []
    slopes_unc = []
    for cdata in datasets:
        npts.append(cdata[0])
        waves.append(cdata[1])
        intercepts.append(cdata[2])
        slopes.append(cdata[3])
        intercepts_unc.append(cdata[4])
        slopes_unc.append(cdata[5])
    all_npts = np.concatenate(npts)
    all_waves = np.concatenate(waves)
    all_intercepts = np.concatenate(intercepts)
    all_intercepts_unc = np.concatenate(intercepts_unc)
    all_slopes = np.concatenate(slopes)
    all_slopes_unc = np.concatenate(slopes_unc)

    if no_weights:
        tnpts = len(all_waves)
        all_intercepts_unc = np.full(tnpts, 1.0)
        all_slopes_unc = np.full(tnpts, 1.0)

    sindxs = np.argsort(all_waves)
    all_waves = all_waves[sindxs]
    all_npts = all_npts[sindxs]
    all_intercepts = all_intercepts[sindxs]
    all_slopes = all_slopes[sindxs]
    all_intercepts_unc = all_intercepts_unc[sindxs]
    all_slopes_unc = all_slopes_unc[sindxs]

    gvals = (all_npts > 0) & (all_waves >= wrange[0]) & (all_waves <= wrange[1])

    fit = LevMarLSQFitter()
    or_fit = FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=5.0)
    rejsym = "kx"

    # intercept
    # cmodelfit = fit(
    #     cmodelinit, 1.0 / all_waves[gvals].value, all_intercepts[gvals], maxiter=500
    # )
    fitx = 1.0 / all_waves[gvals].value
    cmodelfit, mask = or_fit(
        models[0], fitx, all_intercepts[gvals], weights=1.0 / all_intercepts_unc[gvals]
    )
    filtered_data = np.ma.masked_array(all_intercepts[gvals], mask=~mask)
    fitted_models = [cmodelfit]

    np.set_printoptions(precision=5, suppress=True)
    print("intercepts")
    print(cmodelfit.param_names)
    print(repr(cmodelfit.parameters))

    ax[0].plot(all_waves[gvals], cmodelfit(fitx), color="k", alpha=0.5)
    ax[0].plot(all_waves[gvals], filtered_data, rejsym, label="rejected")

    for cdata, ccolor in zip(datasets, colors):
        plot_resid(ax[1], cdata, 2, cmodelfit, ccolor)
    filtered_data2 = np.ma.masked_array(
        all_intercepts[gvals] - cmodelfit(fitx), mask=~mask
    )
    ax[1].plot(all_waves[gvals], filtered_data2, rejsym, label="rejected")

    # slope
    if show_rv:
        cmodelfit, mask = or_fit(
            models[1], fitx, all_slopes[gvals], weights=1.0 / all_slopes_unc[gvals]
        )
        filtered_data = np.ma.masked_array(all_slopes[gvals], mask=~mask)
        fitted_models.append(cmodelfit)

        print("slopes")
        print(cmodelfit.param_names)
        print(repr(cmodelfit.parameters))

        ax[2].plot(all_waves[gvals], cmodelfit(fitx), color="k", alpha=0.5)
        ax[2].plot(all_waves[gvals], filtered_data, rejsym, label="rejected")

        for cdata, ccolor in zip(datasets, colors):
            plot_resid(ax[3], cdata, 3, cmodelfit, ccolor)
        filtered_data2 = np.ma.masked_array(all_slopes[gvals] - cmodelfit(fitx), mask=~mask)
        print("total masked", np.sum(mask))
        ax[3].plot(all_waves[gvals], filtered_data2, rejsym, label="rejected")

    return fitted_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavereg",
        choices=["uv", "opt", "ir", "all", "g25"],
        default="all",
        help="Wavelength region to plot",
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get irv parameters
    path = "/home/kgordon/Python/fuv_mir_rv_relationship/results/"
    gor09_fuse = QTable.read(f"{path}gor09_fuse_irv_params.fits")
    # gor09_iue = QTable.read("results/gor09_iue_irv_params.fits")

    fit19_stis = QTable.read(f"{path}fit19_stis_irv_params.fits")

    # gor21_iue = QTable.read("results/gor21_iue_irv_params.fits")
    gor21_irs = QTable.read(f"{path}gor21_irs_irv_params.fits")

    # dec22_iue = QTable.read("results/dec22_iue_irv_params.fits")
    dec22_spexsxd = QTable.read(f"{path}dec22_spexsxd_irv_params.fits")
    dec22_spexlxd = QTable.read(f"{path}dec22_spexlxd_irv_params.fits")

    # remove UV from Fitzpatrick19 (only want the optical STIS data)
    gvals = fit19_stis["waves"] > 0.30 * u.micron
    fit19_stis = fit19_stis[gvals]

    aiue_iue = QTable.read(f"{path}aiue_iue_irv_params.fits")

    show_rv = False

    # setup plot
    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=3)
    plt.rc("axes", linewidth=3)
    plt.rc("xtick.major", width=3, size=10)
    plt.rc("xtick.minor", width=2, size=5)
    plt.rc("ytick.major", width=3, size=10)
    plt.rc("ytick.minor", width=2, size=5)
    if show_rv:
        nrows = 4
        hratios = [3, 1, 3, 1]
    else:
        nrows=2
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

    gor09_color = "blueviolet"
    fit19_color = "mediumseagreen"
    dec22_color = "darkorange"
    gor21_color = "salmon"
    aiue_color = "royalblue"

    # plot parameters
    yrange_b_type = "linear"
    yrange_s_type = "linear"
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    leg_loc = "lower left"

    leg_loc = "upper right"

    gor09_res1 = plot_irv_ssamp(ax, gor09_fuse, "GCC09", color=gor09_color)
    alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
    fit19_res = plot_irv_ssamp(
        ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
    )
    dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
    dec22_res2 = plot_irv_ssamp(
        ax,
        dec22_spexlxd,
        None,
        color=dec22_color,
        inst="SpeXLXD",
    )
    gor21_res = plot_irv_ssamp(ax, gor21_irs, "G21", color=gor21_color)

    xrange = [0.09, 1.1]
    yrange_a_type = "log"
    yrange_a = [0.0001, 8.0]
    yrange_b = [-1.5, 50.0]
    yrange_s = [0.0, 1.5]
    xticks = [0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3,
                0.35, 0.45, 0.55, 0.7, 0.9, 1.0,
                1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

    # increase the weight of the 2175 A bump region to ensure it is fit well
    # as has been done since FM90
    # done by decreasing the uncdertainties
    #bvals = (alliue_res[1] > 0.20 * u.micron) & (alliue_res[1] < 0.24 * u.micron)
    #alliue_res[4][bvals] /= 5.0
    #alliue_res[5][bvals] /= 5.0

    # decrease the weight of the IUE data in the "poor" flux calibration/stability
    # region - see section 4.2 of Fitzpatrick et al. 2019
    #bvals = alliue_res[1] > 0.27 * u.micron
    #alliue_res[4][bvals] *= 5.0
    #alliue_res[5][bvals] *= 5.0

    #bvals = (gor21_res[1] > 5.0 * u.micron) & (gor21_res[1] < 8.0 * u.micron)
    #gor21_res[4][bvals] /= 5000.0
    #gor21_res[5][bvals] /= 5000.0

    # fitting
    datasets = [alliue_res, gor09_res1, fit19_res, dec22_res1, dec22_res2, gor21_res]
    colors = [aiue_color, gor09_color, fit19_color, dec22_color, dec22_color, gor21_color]
    g25mod = G25()
    #g25mod.FIR_amp = 0.0
    #g25mod.FIR_amp.fixed = True
    fitted_models = plot_wavereg(
        ax,
        [g25mod, g25mod],
        datasets,
        colors,
        wrange=[0.0912, 40.0] * u.micron,
        no_weights=True,
        show_rv=show_rv,
    )
    ax[1].set_ylim(-0.1, 0.1)
    if show_rv:
        ax[3].set_ylim(-5.0, 5.0)

    # plot components
    comps = copy.deepcopy(fitted_models[0])
    modx = np.logspace(np.log10(0.001), np.log10(1000.0), 1000) * u.micron
    ax[0].plot(modx, comps(modx), "k-", alpha=0.75)

    amps = ["bkg", "fuv", "bump", "iss1", "iss2", "iss3", "sil1", "sil2", "fir"]
    for camp in amps:
        setattr(comps, f"{camp}_amp", 0.0)
    for camp in amps:
        setattr(comps, f"{camp}_amp", getattr(fitted_models[0], f"{camp}_amp"))
        ax[0].plot(modx, comps(modx), "k--", alpha=0.5)
        setattr(comps, f"{camp}_amp", 0.0)

    leg_loc = "upper center"
    labels = ["GCC09", "All", "F19", "D22", "G21"]
    label_colors = [gor09_color, aiue_color, fit19_color, dec22_color, gor21_color]
    label_xpos = [0.115, 0.2, 0.5, 2.0, 12.0]
    label_ypos = 10.0
    for clabel, cxpos, ccolor in zip(labels, label_xpos, label_colors):
        ax[0].text(
            cxpos,
            label_ypos,
            clabel,
            color=ccolor,
            bbox=props,
            ha="center",
            fontsize=0.8 * fontsize,
        )

    gor09_res1 = plot_irv_ssamp(ax, gor09_fuse, "GCC09", color=gor09_color)
    alliue_res = plot_irv_ssamp(ax, aiue_iue, "All", color=aiue_color, inst="IUE")
    fit19_res = plot_irv_ssamp(
        ax, fit19_stis, "F19", color=fit19_color, inst="STIS"
    )
    dec22_res1 = plot_irv_ssamp(ax, dec22_spexsxd, "D22", color=dec22_color)
    dec22_res2 = plot_irv_ssamp(
        ax,
        dec22_spexlxd,
        None,
        color=dec22_color,
        inst="SpeXLXD",
    )
    gor21_res = plot_irv_ssamp(ax, gor21_irs, "G21", color=gor21_color)
    xrange = [0.001, 1000.0]
    yrange_a_type = "log"
    yrange_a = [0.001, 20.0]
    yrange_b_type = "symlog"
    yrange_b = [-2.0, 50.0]
    yrange_s_type = "log"
    yrange_s = [0.001, 1.0]
    xticks = [
        0.1,
        0.2,
        0.3,
        0.5,
        0.7,
        1.0,
        2.0,
        3.0,
        5.0,
        7.0,
        10.0,
        20.0,
        30.0,
    ]

    ax[1].set_ylim(-10, 10)
    if show_rv:
        ax[3].set_ylim(-1.0, 1.0)

    # annotate features
    flabels = [
        "Carbonaceous\n2175 " + r"$\AA$",
        "Silicate\n " + r"10 $\mu$m",
        "Silicate\n " + r"20 $\mu$m",
    ]
    fpos = [(0.2175, 0.3), (10.0, 0.15), (20.0, 0.15)]
    for clab, cpos in zip(flabels, fpos):
        ax[0].annotate(
            clab,
            cpos,
            va="bottom",
            ha="center",
            fontsize=0.7 * fontsize,
            alpha=0.7,
            bbox=props,
        )

    custom_lines = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=6),
        Line2D([0], [0], color="black", lw=2, linestyle="solid", alpha=0.5),
    ]

    ax[0].set_yscale(yrange_a_type)
    ax[0].set_ylim(yrange_a)
    ax[0].set_ylabel("intercept (a)")
    ax[1].set_ylabel("(a - fit)/a(unc)")

    nplts = 2
    if show_rv:
        ax[2].legend(custom_lines, ["Data", "Model"], fontsize=fontsize * 0.7)

        # set the wavelength range for all the plots
        ax[2].set_yscale(yrange_b_type)
        ax[2].set_ylim(yrange_b)
        ax[2].set_ylabel("slope (b)")
        ax[3].set_ylabel("b - fit")

        nplts = 4

    for i in range(nplts):
        ax[i].axhline(linestyle="--", alpha=0.25, color="k", linewidth=2)

    # ax[0].legend(ncol=2, loc=leg_loc, fontsize=0.8 * fontsize)

    if show_rv:
        tax = 3
    else:
        tax = 1

    ax[tax].set_xscale("log")
    ax[tax].set_xlim(xrange)
    ax[tax].set_xlabel(r"$\lambda$ [$\mu$m]")

    ax[tax].xaxis.set_major_formatter(ScalarFormatter())
    ax[tax].xaxis.set_minor_formatter(ScalarFormatter())
    ax[tax].set_xticks(xticks, minor=True)

    ax[tax].tick_params(axis="x", which="minor", labelsize=fontsize * 0.8)

    ax[0].yaxis.set_major_formatter(ScalarFormatter())
    ax[0].yaxis.set_minor_formatter(ScalarFormatter())
    ax[0].set_yticks([0.1, 1.0, 10.0], minor=True)
    if show_rv:
        ax[2].yaxis.set_major_formatter(ScalarFormatter())
        ax[2].yaxis.set_minor_formatter(ScalarFormatter())
        ax[2].set_yticks([-1.0, 1.0, 10.0], minor=True)

    fname = f"fuv_mir_irv_g25_fit"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()
