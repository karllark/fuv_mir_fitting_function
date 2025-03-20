import copy
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.fitting import fitter_to_model_params
import astropy.units as u
from astropy.table import QTable

from models_mcmc_extension import EmceeFitter

from dust_extinction.shapes import FM90_B3

# from helpers import G25_FUV as G25
from helpers import G25

from measure_extinction.extdata import ExtData


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("extfile", help="file with extinction curve")
    parser.add_argument(
        "--mcmc",
        help="run the MCMC sampling in addition to minimizer",
        action="store_true",
    )
    parser.add_argument(
        "--nsteps", type=int, default=100, help="# of steps in MCMC chain"
    )
    parser.add_argument(
        "--burnfrac", type=float, default=0.1, help="fraction of MCMC chain to burn"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    file = args.extfile

    # initialize the model
    mod_init = G25()

    # mod_init.bkg_center.fixed = True
    # mod_init.bkg_fwhm.fixed = True

    remove_below_lya = True
    if "gor09" in file:
        remove_below_lya = False

    if "fit19" not in file:
        mod_init.iss1_amp.fixed = True
        mod_init.iss2_amp.fixed = True
        mod_init.iss3_amp.fixed = True

    if "gor21" not in file:
        mod_init.sil1_amp.fixed = True
        mod_init.sil1_center.fixed = True
        mod_init.sil1_fwhm.fixed = True

        mod_init.sil2_amp.fixed = True
        mod_init.sil2_center.fixed = True
        mod_init.sil2_fwhm.fixed = True

        mod_init.fir_amp.fixed = True
        # mod_init.fir_amp.value = 0.15
        mod_init.fir_center.fixed = True
        mod_init.fir_fwhm.fixed = True
    else:
        mod_init.sil2_amp.fixed = False
        mod_init.sil2_center.fixed = True
        mod_init.sil2_fwhm.fixed = True

        mod_init.fir_center.fixed = True
        mod_init.fir_fwhm.fixed = True

    if "dec22" in file:
        mod_init.fir_amp.fixed = True
        mod_init.fir_center.fixed = True
        mod_init.fir_fwhm.fixed = True

    # get a saved extinction curve
    ofile = file.replace(".fits", "_fit.fits").replace("data/", "fits/")

    ext = ExtData(filename=file)
    if "IUE" in ext.exts.keys():
        spectype = "IUE"
    else:
        spectype = "STIS"

    # get the extinction curve in alav - (using (A(V) in header)
    if ext.type == "elx":
        ext.trans_elv_alav()

    wave, y, y_unc = ext.get_fitdata(
        remove_uvwind_region=True,
        remove_below_lya=remove_below_lya,
        remove_lya_region=True,
    )

    if "dec22" in file:
        gvals = wave < 4.0 * u.micron
        wave = wave[gvals]
        y = y[gvals]
        y_unc = y_unc[gvals]

    x = 1.0 / wave.value

    # modify weights to make sure the 2175 A bump is fit
    weights = 1.0 / y_unc
    # weights = np.full(len(x), 1.0 / (0.1 * y))
    # weights[(x > 4.0) & (x < 5.1)] *= 10.0
    # weights[(x > 1/0.13) & (x < 0.15)] *= 10.0
    # weights[x > 1./0.12] *= 0.0

    # FM90 only applies to the UV
    mod_init_fm90 = FM90_B3()
    gvals_fm90 = x > 1 / 0.3

    if np.sum(gvals_fm90) > 0:
        fit_fm90 = True
    else:
        fit_fm90 = False

    # fit the data to the model using the fitter
    fit = LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fitmod = fit(mod_init, x, y, weights=weights)
        plotmod = fitmod

        if fit_fm90:
            fitmod_fm90 = fit(
                mod_init_fm90, x[gvals_fm90], y[gvals_fm90], weights=weights[gvals_fm90]
            )
            plotmod_fm90 = fitmod_fm90

    # sample with optimizer
    if args.mcmc:
        fitmcmc = EmceeFitter(
            nsteps=args.nsteps,
            burnfrac=args.burnfrac,
            save_samples=ofile.replace(".fits", ".h5"),
        )
        fitmod_mcmc = fitmcmc(fitmod, x, y, weights=weights)
        plotmod = fitmod_mcmc

        if fit_fm90:
            fitmcmc_fm90 = EmceeFitter(
                nsteps=args.nsteps,
                burnfrac=args.burnfrac,
                save_samples=ofile.replace(".fits", "_fm90.h5"),
            )
            fitmod_mcmc_fm90 = fitmcmc_fm90(fitmod, x, y, weights=weights)
            plotmod_fm90 = fitmod_mcmc_fm90

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fitmcmc.plot_emcee_results(
                fitmod_mcmc, filebase=ofile.replace(".fits", "_mcmc")
            )
            if fit_fm90:
                fitmcmc_fm90.plot_emcee_results(
                    fitmod_mcmc_fm90, filebase=ofile.replace(".fits", "fm90_mcmc")
                )

        # diagnostics
        print(
            "G25 autocorr tau = ",
            fitmcmc.fit_info["sampler"].get_autocorr_time(quiet=True),
        )

        g25_param = ext.create_param_table(
            fitmod_mcmc.param_names,
            fitmod_mcmc.parameters,
            parameters_punc=fitmod_mcmc.uncs_plus,
            parameters_munc=fitmod_mcmc.uncs_minus,
        )
        if fit_fm90:
            fm90_param = ext.create_param_table(
                fitmod_mcmc_fm90.param_names,
                fitmod_mcmc_fm90.parameters,
                parameters_punc=fitmod_mcmc_fm90.uncs_plus,
                parameters_munc=fitmod_mcmc_fm90.uncs_minus,
            )

        print("G25 parameters (p50)")
        fitmod_mcmc.pprint_parameters()
    else:
        g25_param = ext.create_param_table(fitmod.param_names, fitmod.parameters)
        if fit_fm90:
            fm90_param = ext.create_param_table(
                fitmod_fm90.param_names, fitmod_fm90.parameters
            )

        print("G25 parameters (best)")
        fitmod.pprint_parameters()

    # save extinction and fit parameters
    fit_params = {}
    fit_params["G25"] = g25_param
    if fit_fm90:
        fit_params["FM90"] = fm90_param
    if fit_fm90:
        chisqr_fm90 = np.sum(
            np.square((y[gvals_fm90] - plotmod_fm90(x[gvals_fm90])) / y_unc[gvals_fm90])
        ) / np.sqrt(np.sum(gvals_fm90) - 1)
        chisqr_fm90_g25 = np.sum(
            np.square((y[gvals_fm90] - plotmod(x[gvals_fm90])) / y_unc[gvals_fm90])
        ) / np.sqrt(np.sum(gvals_fm90) - 1)
        ctab = QTable()
        ctab["name"] = ["G25", "other"]
        ctab["FM90"] = [chisqr_fm90_g25, chisqr_fm90]
        fit_params["CHISQR"] = ctab
        print("FM90 chisqr (G25, FM90)", chisqr_fm90_g25, chisqr_fm90)

    ext.save(ofile, fit_params=fit_params)

    fontsize = 16
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=3)
    plt.rc("xtick.major", width=3, size=10)
    plt.rc("xtick.minor", width=2, size=5)
    plt.rc("ytick.major", width=3, size=10)
    plt.rc("ytick.minor", width=2, size=5)

    # plot the observed data, initial guess, and final fit
    fig, fax = plt.subplots(
        nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    # remove pesky x without units warnings
    x /= u.micron
    x = 1 / x  # covert to microns

    ax = fax[0]
    ax.plot(x, y, "k-", label="Observed Data")
    ax.plot(x, y + y_unc, "k:")
    ax.plot(x, y - y_unc, "k:")

    modx = np.logspace(np.log10(0.0912), np.log10(32.0), num=1000) * u.micron
    modgvals_fm90 = modx < 0.3 * u.micron
    # ax.plot(x, fm90_fit3(x), label="emcee")
    if args.mcmc:
        ptype = "P50"

        # plot samples from the mcmc chain
        flat_samples = fitmcmc.fit_info["sampler"].get_chain(
            discard=int(args.burnfrac * args.nsteps), flat=True
        )
        inds = np.random.randint(len(flat_samples), size=100)
        model_copy = fitmod_mcmc.copy()
        for ind in inds:
            sample = flat_samples[ind]
            fitter_to_model_params(model_copy, sample)
            ax.plot(x, model_copy(x), "C1", alpha=0.05)
    else:
        ptype = "Best"

    ax.plot(modx, plotmod(modx), "g-", alpha=0.75, label=f"{ptype}")
    if fit_fm90:
        ax.plot(
            modx[modgvals_fm90],
            plotmod_fm90(modx[modgvals_fm90]),
            "b:",
            alpha=0.5,
            label=f"{ptype} (FM90)",
        )

    # plot the components
    amps = ["bkg", "fuv", "bump", "iss1", "iss2", "iss3", "sil1", "sil2", "fir"]
    comps = copy.deepcopy(plotmod)
    for camp in amps:
        if camp not in ["bkg", "fir"]:
            setattr(comps, f"{camp}_amp", 0.0)
    for camp in amps:
        setattr(comps, f"{camp}_amp", getattr(fitmod, f"{camp}_amp"))
        ax.plot(modx, comps(modx), "g--", alpha=0.5)
        if camp not in ["bkg", "fir"]:
            setattr(comps, f"{camp}_amp", 0.0)

    krange = [np.nanmin(x.value), np.nanmax(x.value)]
    ax.set_xlim(krange)
    krange = [np.max((np.nanmin(y), 1e-3)), np.nanmax(y)]
    ax.set_ylim(krange)

    ax.set_xscale("log")
    ax.set_ylabel(r"$A(\lambda)/A(V)$")
    ax.set_yscale("log")

    ax.set_title(file)

    ax.legend(loc="best")

    # residuals
    ax = fax[1]
    ax.plot(x, np.zeros((len(x))), "k--")
    if fit_fm90:
        ax.plot(
            x[gvals_fm90],
            (y[gvals_fm90] - plotmod_fm90(x[gvals_fm90])) / y_unc[gvals_fm90],
            "b:",
            alpha=0.5,
        )
    ax.plot(x, (y - plotmod(x)) / y_unc, "g-", alpha=0.75)
    ax.set_ylim(np.array([-1.0, 1.0]) * 3.0)
    ax.set_xlabel(r"$\lambda$ [$\mu m$]")
    ax.set_ylabel("(y - mod)/unc")

    plt.tight_layout()

    # plot or save to a file
    outname = ofile.replace(".fits", "")
    if args.png:
        fig.savefig(outname + ".png")
    elif args.pdf:
        fig.savefig(outname + ".pdf")
    else:
        plt.show()
