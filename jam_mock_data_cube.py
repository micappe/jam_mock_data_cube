"""
    Copyright (C) 2024-2025, Michele Cappellari

    E-mail: michele.cappellari_at_physics.ox.ac.uk

    Updated versions of the software are available from my web page
    https://github.com/micappe/jam_mock_data_cube

    If you have found this software useful for your research,
    we would appreciate a reference to the paper
    "Nguyen D.D. et al. (submitted)".

    Changelog
    ---------
    V1.1.0: Rewritten dosctrings. MC, Oxford, 1 June 2025
    V1.0.0: 12 June 2024: Initial version.

"""

import numpy as np
from scipy import special

import ppxf.ppxf_util as util
from jampy.axi.jam_axi_proj import jam_axi_proj, mge_surf
from mgefit.mge_fit_1d import mge_fit_1d

##############################################################################

def sersic_to_mge(mag_tot, re, n, q, mag_sun):
    """
    Convert a Sersic profile to a Multi-Gaussian Expansion (MGE).

    This function takes Sersic parameters, generates the 1D Sersic profile,
    and fits it with an MGE using `mge_fit_1d`.

    """
    # Generate Sersic profile
    radius = np.logspace(-3, np.log10(10 * re), 300)
    b_n = special.gammainccinv(2 * n, 0.5)
    integ = np.pi * np.exp(b_n) * special.gamma(1 + 2 * n) / b_n**(2 * n)
    mu_e = mag_tot + 2.5 * np.log10(integ * re**2)
    mu_r = mu_e + 2.5 * b_n / np.log(10) * ((radius / re)**(1 / n) - 1)
    surf_sersic = (64800 / np.pi)**2 * 10**(0.4 * (mag_sun - mu_r))

    # Fit Sersic profile with MGE
    w = surf_sersic > surf_sersic[0]/1e40  # samples at most 40 orders of magnitude in density
    surf, sigma = mge_fit_1d(radius[w], surf_sersic[w], ngauss=20, plot=False).sol
    qobs = np.full_like(sigma, q)

    return surf, sigma, qobs

##############################################################################

class JAMCubeSimulator:
    """
    Create a 3D spectral data cube using MGE and JAM model parameters.

    The surface brightness profile can be specified either as a Multi-Gaussian
    Expansion (MGE) or as a Sersic profile. The gravitational potential can be
    set by providing either the mass-to-light ratio (M/L), the total galaxy
    mass, and/or the black hole mass.

    Given a calibrated MGE or Sersic surface brightness profile and an input
    template spectrum, this class generates a spectral data cube (nspec, ny,
    nx) and a 2D array of spectra (nspec, npix) for a simulated observation.
    Kinematics are derived using the Jeans Anisotropic Modelling (JAM) method.

    The input redshift is used to shift the template spectrum before convolving
    it with the JAM kinematics and resampling it onto the output wavelength grid.
    The MGE surface brightness, assumed to be the observed surface brightness
    (not intrinsic, no cosmological dimming applied by this routine), is used
    to scale the spectrum at each spatial pixel. The resulting mock data cube
    is by design in observed flux density units (erg/s/cm^2/Angstrom/arcsec^2)
    and does not require any further cosmological dimming correction.

    Parameters
    ----------
    profile : dict
        A dictionary describing the surface brightness profile.
        It must contain a ``type`` key, which can be ``"MGE"`` or ``"Sersic"``.
        - If ``type="MGE"``, the dictionary must contain:
            - ``surf``: MGE surface brightness in Lsun/pc^2 (Ngauss,).
            - ``sigma``: MGE sigma in arcsec (Ngauss,).
            - ``qobs``: MGE observed flattening (Ngauss,).
        - If ``type="Sersic"``, the dictionary must contain:
            - ``mag_tot``: Total apparent magnitude of the Sersic model.
            - ``re``: Effective (projected half-light) radius in arcsec.
            - ``n``: Sersic index.
            - ``q``: Observed axis ratio.
    inc : float
        Model inclination in degrees (0 for face-on, 90 for edge-on).
    mass_bh : float
        Black hole mass in solar masses.
    distance : float
        Angular-diameter distance to the object in Mpc.
    band_mge : str
        Photometric band in which the MGE surface brightness `surf` is defined
        (e.g., 'V', 'R', 'F814W'). This is used to determine `mag_sun` for scaling.
    lam : array_like
        1D array of wavelengths for the input template spectrum in Angstroms (rest-frame).
    spectrum : array_like
        1D array of flux values for the input template spectrum. The flux scale
        is arbitrary as it will be normalized by the MGE surface brightness
        and `mag_sun`.
    out_fov : float
        Side length of the square field of view for the output cube in arcsec.
    out_pix : float
        Pixel size for the output cube in arcsec/pixel.
    out_lam : array_like
        1D array of desired output wavelengths for the spectra in Angstroms
        (observed-frame). This grid can be non-uniformly sampled.
    redshift : float
        Object redshift. Used to redshift the input template spectrum `lam` and for
        calculating the Sun's magnitude in the `band_mge` at this redshift.
    system : str, optional
        Photometric system for the input spectrum ('AB' or 'Vega').
        Default is 'AB'. This affects the calculation of `mag_sun` and
        subsequently `mag_mge`.
    verbose : bool, optional
        If True, print information regarding output resolution and other
        diagnostics during processing. Default is True.
    **kwargs
        Additional keyword arguments passed directly to
        `jampy.axi.jam_axi_proj.jam_axi_proj` for the kinematic calculations.
        Common examples include `beta` (anisotropy), `gamma` (tangential anisotropy),
        `align` (ellipsoid alignment, 'cyl' or 'sph'), `analytic_los` (bool),
        `kappa` (rotation scaling).

    Attributes
    ----------
    spectra : ndarray
        2D array of output spectra, with shape (nspec, npix).
        Units: erg/s/cm^2/Angstrom/arcsec^2.
        `nspec` is the number of wavelength points (`out_lam.size`),
        `npix` is the total number of spatial pixels in the FOV.
    cube : ndarray
        3D data cube of output spectra, with shape (nspec, ny, nx).
        Units: erg/s/cm^2/Angstrom/arcsec^2.
        This is a reshaped version of `spectra`. `ny` and `nx` are the
        spatial dimensions of the output grid, determined by `out_fov`
        and `out_pix`.
    lam : ndarray
        1D array of output wavelengths for the spectra in Angstroms (observed-frame).
        This is the same as the input `out_lam`.
    band_mge : str
        Photometric band of the MGE, as provided in the `band_mge` parameter.
    mag_sun : float
        Apparent magnitude of the Sun in the `band_mge` photometric band,
        as if observed at the given `redshift` and using the specified `system`.
        Used for flux scaling.
    mag_mge : ndarray
        2D array of the MGE model's surface brightness in mag/arcsec^2
        at each (x, y) pixel of the output grid. Shape (ny, nx), where
        ny and nx are the spatial dimensions of the cube.
    surf_band : ndarray
        2D array of the MGE model's surface brightness in Lsun/pc^2
        (observed) at each (x, y) pixel of the output grid. Shape (ny, nx).
    vel : ndarray
        1D array of mean line-of-sight velocities (km/s) from the JAM model
        at each of the `npix` spatial pixels. This is before convolution with
        the spectral line shape.
    rms : ndarray
        1D array of root-mean-square (RMS) line-of-sight velocities (km/s)
        from the JAM model at each of the `npix` spatial pixels.
    sig : ndarray
        1D array of line-of-sight velocity dispersions (sigma, km/s)
        from the JAM model at each of the `npix` spatial pixels.
        Calculated as `sqrt(rms**2 - vel**2)`.
    x : ndarray
        1D array of x-coordinates (arcsec) for the center of each spatial
        pixel in the flattened `spectra` array (and for `vel`, `rms`, `sig`).
        Length `npix`. The origin (0,0) is at the center of the FOV.
    y : ndarray
        1D array of y-coordinates (arcsec) for the center of each spatial
        pixel in the flattened `spectra` array (and for `vel`, `rms`, `sig`).
        Length `npix`. The origin (0,0) is at the center of the FOV.

    Examples
    --------
    Create a mock data cube for a galaxy observation:

    >>> import numpy as np
    >>> from jam_mock_data_cube import jam_mock_data_cube
    >>>
    >>> # Example MGE parameters
    >>> profile_mge = {
    ...     "type": "MGE",
    ...     "surf": np.array([1000., 500.]),  # Lsun/pc^2 (observed)
    ...     "sigma": np.array([1.0, 3.0]),     # arcsec
    ...     "qobs": np.array([0.9, 0.7])
    ... }
    >>>
    >>> # Example template spectrum
    >>> lam_spec_template = np.linspace(4000, 7000, 1000)  # Angstroms (rest-frame)
    >>> flux_spec_template = np.exp(-((lam_spec_template - 5500)/200)**2)
    >>>
    >>> # Example output parameters
    >>> output_wavelengths = np.linspace(4500, 6500, 500)  # Angstroms (observed-frame)
    >>>
    >>> # Create the mock data cube
    >>> mock = jam_mock_data_cube(
    ...     profile=profile_mge,
    ...     inc=60.,              # degrees
    ...     mass_bh=1e8,          # Msun
    ...     distance=20.,         # Mpc
    ...     band_mge='V',         # Photometric band of MGE surf
    ...     lam=lam_spec_template,
    ...     spectrum=flux_spec_template,
    ...     out_fov=10.,          # arcsec
    ...     out_pix=0.1,          # arcsec/pixel
    ...     out_lam=output_wavelengths,
    ...     redshift=0.05,
    ...     system='AB',
    ...     verbose=True
    ... )
    >>>
    >>> # Access the results
    >>> data_cube_3d = mock.cube
    >>> spectra_2d_array = mock.spectra
    >>> final_wavelength_grid = mock.lam
    >>> velocity_field_1d = mock.vel    
    
    References
    ----------
    If you have found this software useful for your research,
    we would appreciate a reference to the paper:
    `Nguyen D.D., et al. (2026), MNRAS, 546, stag238 <https://ui.adsabs.harvard.edu/abs/2026MNRAS.546ag238N>`_

    The kinematic calculations rely on the Jeans Anisotropic Modelling (JAM) method, described in:
    `Cappellari, M. (2008), MNRAS, 390, 71 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_
    `Cappellari, M. (2020), MNRAS, 494, 4819 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C>`_

    The Multi-Gaussian Expansion (MGE) fitting procedures are implemented in:
    `Cappellari, M. (2002), MNRAS, 333, 400 <https://ui.adsabs.harvard.edu/abs/2002MNRAS.333..400C>`_
    """
    def __init__(self, profile, inc, mass_bh, distance, band, lam, spectrum, out_fov, out_pix, out_lam, redshift,
                 ml=None, mass_gal=None, system='AB', verbose=True, analytic_los=0, **kwargs):

        mag_sun = util.mag_sun(band, redshift, system=system)

        if profile['type'] == 'MGE':
            surf_lum = profile['surf']
            sigma = profile['sigma']
            qobs = profile['qobs']
        elif profile['type'] == 'Sersic':
            mag_tot = profile['mag_tot']
            re = profile['re']
            n = profile['n']
            q = profile['q']
            surf_lum, sigma, qobs = sersic_to_mge(mag_tot, re, n, q, mag_sun)
        else:
            raise ValueError("Profile type must be 'MGE' or 'Sersic'")

        # --- Compute mass surface density for potential ---
        if ml is not None:
            surf_pot = surf_lum * ml  # MGE surface brightness * M/L
        elif mass_gal is not None:
            # Compute total luminosity of MGE
            pc = distance*np.pi/0.648  # Factor to convert arcsec --> pc (with distance in Mpc)
            L_tot = 2 * np.pi * np.sum(surf_lum * (sigma*pc)**2 * qobs)
            ml = mass_gal / L_tot
            surf_pot = surf_lum * ml
        else:
            # Default: assume M/L=1
            surf_pot = surf_lum

        nx = int(np.ceil(out_fov/2/out_pix))
        xx = np.linspace(0.5 - nx, nx - 0.5, 2*nx)*out_pix
        x, y = map(np.ravel, np.meshgrid(xx, xx))

        # Use surf_lum as first parameter, surf_pot as fourth parameter
        vel = jam_axi_proj(surf_lum, sigma, qobs, surf_pot, sigma, qobs, inc, mass_bh, distance,
                           x, y, moment='z', analytic_los=analytic_los, **kwargs).model
        rms = jam_axi_proj(surf_lum, sigma, qobs, surf_pot, sigma, qobs, inc, mass_bh, distance, 
                           x, y, moment='zz', analytic_los=analytic_los, **kwargs).model
        sig = np.sqrt(rms**2 - vel**2)

        # Bring spectrum to desired redshift
        lam = lam*(1 + redshift)

        # Use full wave range as the MGE band may be different from IFS band
        # NB: `system` is only used to output `mag_mge` in the requested system
        mag_spectrum = util.mag_spectrum(lam, spectrum, bands=band, system=system)

        # Truncate spectrum to desired observed wavelength range
        good = (lam > out_lam.min()) & (lam < out_lam.max())
        lam, spectrum = lam[good], spectrum[good]

        surf_band = mge_surf(x, y, surf_lum, sigma, qobs)   # Lsun/pc^2 of MGE

        # Surface brightness of every (x,y) pixel in mag/arcsec^2
        # eq.(3) of readme_mge_fit_sectors.pdf here https://pypi.org/project/mgefit/
        mag_mge = mag_sun - 2.5*np.log10(surf_band*(np.pi/64800)**2)  

        c = 299792.458                          # speed of light in km/s
        spectra = np.empty((out_lam.size, x.size))

        for j, (velj, sigj) in enumerate(zip(vel, sig)):
            sig_x = lam*sigj/c   # sigma in Angstroms of every x-coordinate
            spectrum_conv = util.varsmooth(lam*(1 + velj/c), spectrum, sig_x, out_lam)
            spectra[:, j] = spectrum_conv*10**(0.4*(mag_spectrum - mag_mge[j]))

        if verbose:
            rat = np.median(np.diff(lam))/np.median(np.diff(out_lam))
            if rat > 1:
                print(f"WARNING: provided input spectrum has resolution"
                      f" {rat:.1f}x lower than requested output one")
            print(f"--> np.median(np.diff(lam))/np.median(np.diff(out_lam)): {rat:.1f}")

        self.spectra = spectra          # Stack of `spectra` (npix, nspec)
        self.cube = spectra.reshape(out_lam.size, xx.size, xx.size)  # cube of spectra (npix, nx, ny)
        self.lam = out_lam              # wavelength of the `spectra`
        self.band_mge = band
        self.mag_sun = mag_sun
        self.mag_mge = mag_mge          # mag/arcsec^2 at every (x, y)
        self.surf_band = surf_band      # Lsun/pc^2 at every (x, y)
        self.vel = vel                  # Kinematics from JAM model
        self.rms = rms
        self.sig = sig
        self.x = x                      # Coordinates of the `spectra`
        self.y = y

##############################################################################
