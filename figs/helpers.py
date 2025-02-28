import numpy as np
import astropy.units as u

import warnings

from astropy.modeling import Fittable1DModel, Parameter


# from dust_extinction.shapes import G21
from dust_extinction.baseclasses import BaseExtModel
from dust_extinction.shapes import _modified_drude

x_range_G22 = [1.0 / 45.0, 1.0 / 0.08]


class SpectralUnitsWarning(UserWarning):
    pass


def _get_x_in_wavenumbers(in_x):
    """
    Convert input x to wavenumber given x has units.
    Otherwise, assume x is in waveneumbers and issue a warning to this effect.
    Parameters
    ----------
    in_x : astropy.quantity or simple floats
        x values
    Returns
    -------
    x : floats
        input x values in wavenumbers w/o units
    """
    # handles the case where x is a scaler
    in_x = np.atleast_1d(in_x)

    # check if in_x is an astropy quantity, if not issue a warning
    if not isinstance(in_x, u.Quantity):
        warnings.warn(
            "x has no units, assuming x units are inverse microns", SpectralUnitsWarning
        )

    # convert to wavenumbers (1/micron) if x input in units
    # otherwise, assume x in appropriate wavenumber units
    with u.add_enabled_equivalencies(u.spectral()):
        x_quant = u.Quantity(in_x, 1.0 / u.micron, dtype=np.float64)

    # strip the quantity to avoid needing to add units to all the
    #    polynomical coefficients
    return x_quant.value


class G25(BaseExtModel, Fittable1DModel):
    r"""
    Gordon (2024) parameter model.  Inspired by Pei (1992), Massa et al. (2020), and Gordon et al. (2021)

    Parameters
    ----------
    BKG_amp : float
      background term amplitude
    BKG_lambda : float
      background term central wavelength
    BKG_b : float
      background term b coefficient
    BKG_n : float
      background term n coefficient [FIXED at n = 2]

    FUV_amp : float
      far-ultraviolet term amplitude
    FUV_lambda : float
      far-ultraviolet term central wavelength
    FUV_b : float
      far-ultraviolet term b coefficent
    FUV_n : float
      far-ultraviolet term n coefficient

    NUV_amp : float
      near-ultraviolet (2175 A) term amplitude
    NUV_lambda : float
      near-ultraviolet (2175 A) term central wavelength
    NUV_b : float
      near-ultraviolet (2175 A) term b coefficent
    NUV_n : float
      near-ultraviolet (2175 A) term n coefficient [FIXED at n = 2]

    SIL1_amp : float
      1st silicate feature (~10 micron) term amplitude
    SIL1_lambda : float
      1st silicate feature (~10 micron) term central wavelength
    SIL1_b : float
      1st silicate feature (~10 micron) term b coefficent
    SIL1_n : float
      1st silicate feature (~10 micron) term n coefficient [FIXED at n = 2]

    SIL2_amp : float
      2nd silicate feature (~18 micron) term amplitude
    SIL2_lambda : float
      2nd silicate feature (~18 micron) term central wavelength
    SIL2_b : float
      2nd silicate feature (~18 micron) term b coefficient
    SIL2_n : float
      2nd silicate feature (~18 micron) term n coefficient [FIXED at n = 2]

    FIR_amp : float
      far-infrared term amplitude
    FIR_lambda : float
      far-infrared term central wavelength
    FIR_b : float
      far-infrared term b coefficent
    FIR_n : float
      far-infrared term n coefficient [FIXED at n = 2]

    Notes
    -----
    From Pei (1992, ApJ, 395, 130)

    Applicable from the extreme UV to far-IR

    Example showing a P92 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.shapes import P92

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(-3.0, 3.0, num=1000)
        x = (1.0/lam)/u.micron

        ext_model = P92()
        ax.plot(1/x,ext_model(x),label='total')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG only')

        ext_model = P92(NUV_amp=0.0,
                        SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FUV only')

        ext_model = P92(FUV_amp=0.,
                        SIL1_amp=0.0, SIL2_amp=0.0, FIR_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+NUV only')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FIR+SIL1 only')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL1_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FIR+SIL2 only')

        ext_model = P92(FUV_amp=0., NUV_amp=0.0,
                        SIL1_amp=0.0, SIL2_amp=0.0)
        ax.plot(1./x,ext_model(x),label='BKG+FIR only')

        # Milky Way observed extinction as tabulated by Pei (1992)
        MW_x = [0.21, 0.29, 0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
                2.27, 2.50, 2.91, 3.65, 4.00, 4.17, 4.35, 4.57, 4.76,
                5.00, 5.26, 5.56, 5.88, 6.25, 6.71, 7.18, 7.60,
                8.00, 8.50, 9.00, 9.50, 10.00]
        MW_x = np.array(MW_x)
        MW_exvebv = [-3.02, -2.91, -2.76, -2.58, -2.23, -1.60, -0.78, 0.00,
                     1.00, 1.30, 1.80, 3.10, 4.19, 4.90, 5.77, 6.57, 6.23,
                     5.52, 4.90, 4.65, 4.60, 4.73, 4.99, 5.36, 5.91,
                     6.55, 7.45, 8.45, 9.80, 11.30]
        MW_exvebv = np.array(MW_exvebv)
        Rv = 3.08
        MW_axav = MW_exvebv/Rv + 1.0
        ax.plot(1./MW_x, MW_axav, 'o', label='MW Observed')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylim(1e-3,10.)

        ax.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """

    n_inputs = 1
    n_outputs = 1

    bkg_amp = Parameter(
        description="bkg: amplitude", default=2.5, bounds=(0.0, 10.0)
    )
    bkg_center = Parameter(
        description="bkg: center", default=0.047, bounds=(0.03, 0.07), fixed=True
    )
    bkg_fwhm = Parameter(description="bkg: fwhm", default=0.433, fixed=False)

    fuv_amp = Parameter(
       description="fuv: amplitude", default=3.5, bounds=(0.0, 10.0), fixed=False
    )
    fuv_center = Parameter(description="fuv: center", default=0.075, fixed=False)
    fuv_fwhm = Parameter(description="fuv: fwhm", default=0.015, fixed=False)

    #FUV_amp = Parameter(description="FUV term: amplitude", default=14.0 * AbAv, min=0.0)
    #FUV_lambda = Parameter(
    #    description="FUV term: center wavelength",
    #    default=0.075,
    #    bounds=(0.06, 0.10),
    #    fixed=True,
    #)
    #FUV_b = Parameter(description="FUV term: b coefficient", default=4.0)
    #FUV_n = Parameter(description="FUV term: n coefficient", default=6.5)

    bump_amp = Parameter(
        description="bump: amplitude", default=3.23, bounds=(-1.0, 6.0)
    )
    bump_center = Parameter(
        description="bump: centroid", default=4.59, bounds=(4.5, 4.9)
    )
    bump_fwhm = Parameter(description="bump: width", default=0.95, bounds=(0.6, 1.7))

    iss1_amp = Parameter(
        description="ISS1: amplitude", default=0.03893, bounds=(0.001, 1.0), fixed=False
    )
    iss1_center = Parameter(description="ISS1: center", default=2.288, fixed=True)
    iss1_fwhm = Parameter(description="ISS2: fwhm", default=0.243, fixed=True)

    iss2_amp = Parameter(
        description="ISS2: amplitude", default=0.02965, bounds=(0.001, 0.1), fixed=False
    )
    iss2_center = Parameter(description="ISS2: center", default=2.054, fixed=True)
    iss2_fwhm = Parameter(description="ISS2: fwhm", default=0.179, fixed=True)

    iss3_amp = Parameter(
        description="ISS3: amplitude", default=0.01747, bounds=(0.001, 0.1), fixed=False
    )
    iss3_center = Parameter(description="ISS3: center", default=1.587, fixed=True)
    iss3_fwhm = Parameter(description="ISS3: fwhm", default=0.243, fixed=True)

    sil1_amp = Parameter(
        description="silicate 10um: amplitude", default=0.067, bounds=(0.001, 0.3)
    )
    sil1_center = Parameter(
        description="silicate 10um: center", default=9.84, bounds=(8.0, 12.0)
    )
    sil1_fwhm = Parameter(
        description="silicate 10um: fwhm", default=2.21, bounds=(1.0, 3.0)
    )
    sil1_asym = Parameter(
        description="silicate 10um: asymmetry",
        default=-0.25,
        # default=0.0,
        bounds=(-2.0, 2.0),
        fixed=True,
    )
    sil2_amp = Parameter(
        description="silicate 20um: amplitude", default=0.027, bounds=(0.001, 0.3)
    )
    sil2_center = Parameter(
        description="silicate 20um: center",
        default=19.6,
        bounds=(16.0, 24.0),
        fixed=True,
    )
    sil2_fwhm = Parameter(
        description="silicate 20um: fwhm",
        default=7.0,
        bounds=(5.0, 15.0),
        fixed=False,
    )
    sil2_asym = Parameter(
        description="silicate 20um: asymmetry",
        default=-0.27,
        # default=0.0,
        bounds=(-2.0, 2.0),
        fixed=True,
    )

    fir_amp = Parameter(
        description="fir: amplitude", default=0.005, bounds=(0.0, 0.01), fixed=False
    )
    fir_center = Parameter(
        description="fir: center",
        default=15.,
        bounds=(4.0, 100.0),
        fixed=False,
    )
    fir_fwhm = Parameter(
        description="fir: fwhm",
        default=10.0,
        bounds=(1.0, 100.0),
        fixed=False,
    )

    x_range = [1.0 / 1e3, 1.0 / 1e-3]

    @staticmethod
    def _p92_single_term(in_lambda, amplitude, cen_wave, b, n):
        r"""
        Function for calculating a single P92 term

        .. math::

           \frac{a}{(\lambda/cen_wave)^n + (cen_wave/\lambda)^n + b}

        when n = 2, this term is equivalent to a Drude profile

        Parameters
        ----------
        in_lambda: vector of floats
           wavelengths in same units as cen_wave

        amplitude: float
           amplitude

        cen_wave: flaot
           central wavelength

        b : float
           b coefficient

        n : float
           n coefficient
        """
        l_norm = in_lambda / cen_wave

        return amplitude / (np.power(l_norm, n) + np.power(l_norm, -1 * n) + b)

    def evaluate(
        self,
        x,
        bkg_amp,
        bkg_center,
        bkg_fwhm,
        # BKG_amp,
        # BKG_lambda,
        # BKG_b,
        fuv_amp,
        fuv_center,
        fuv_fwhm,
        #FUV_amp,
        #FUV_lambda,
        #FUV_b,
        #FUV_n,
        bump_amp,
        bump_center,
        bump_fwhm,
        iss1_amp,
        iss1_center,
        iss1_fwhm,
        iss2_amp,
        iss2_center,
        iss2_fwhm,
        iss3_amp,
        iss3_center,
        iss3_fwhm,
        sil1_amp,
        sil1_center,
        sil1_fwhm,
        sil1_asym,
        sil2_amp,
        sil2_center,
        sil2_fwhm,
        sil2_asym,
        fir_amp,
        fir_center,
        fir_fwhm,
    ):
        """
        P92 function

        Parameters
        ----------
        x: float
           expects either x in units of wavelengths or frequency
           or assumes wavelengths in wavenumbers [1/micron]

           internally wavenumbers are used

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input x values outside of defined range
        """
        # calculate the terms
        lam = 1.0 / x
        axav = (
            +_modified_drude(lam, bkg_amp, bkg_center, bkg_fwhm, 0.0)
            # + self._p92_single_term(lam, BKG_amp, BKG_lambda, BKG_b, 2.0)
            # + self._p92_single_term(lam, FUV_amp, FUV_lambda, FUV_b, FUV_n)
            + _modified_drude(lam, fuv_amp, fuv_center, fuv_fwhm, 0.0)
            + _modified_drude(x, bump_amp, bump_center, bump_fwhm, 0.0)
            + _modified_drude(x, iss1_amp, iss1_center, iss1_fwhm, 0.0)
            + _modified_drude(x, iss2_amp, iss2_center, iss2_fwhm, 0.0)
            + _modified_drude(x, iss3_amp, iss3_center, iss3_fwhm, 0.0)
            + _modified_drude(lam, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
            + _modified_drude(lam, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)
            + _modified_drude(lam, fir_amp, fir_center, fir_fwhm, 0.0)
        )

        # return A(x)/A(V)
        return axav

    # use numerical derivaties (need to add analytic)
    fit_deriv = None
