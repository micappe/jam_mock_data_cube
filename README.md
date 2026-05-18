# jam_mock_data_cube

A Python tool to generate realistic 3D Integrated Field Spectroscopy (IFS) mock data cubes for galaxies using Multi-Gaussian Expansion (MGE) or Sersic surface brightness profiles combined with Jeans Anisotropic Modelling (JAM) kinematics.

## Features

* **Flexible Profiles:** Accepts galaxy surface brightness inputs parameterized either directly via MGE components or as a 1D Sersic profile (automatically converted to MGE).
* **JAM Kinematics:** Computes line-of-sight velocity ($v$), root-mean-square velocity ($v_{\text{rms}}$), and velocity dispersion ($\sigma$) fields using the Axisymmetric Jeans Anisotropic Modelling method.
* **Realistic Spectral Synthesis:** Shifts an input rest-frame template spectrum according to the object's redshift and local line-of-sight velocity, applies velocity dispersion smoothing using `ppxf` utilities, and outputs fluxes calibrated in standard observed density units ($\text{erg/s/cm}^2/\text{Å}/\text{arcsec}^2$).
* **Flexible Gridding:** Generates an arbitrary user-defined spatial field of view (FOV) and wavelength grid.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install numpy scipy ppxf jampy mgefit

```

## Quick Start

```python
import numpy as np
from jam_mock_data_cube import JAMCubeSimulator

# Define an MGE profile
profile_mge = {
    "type": "MGE",
    "surf": np.array([1000., 500.]),  # Lsun/pc^2 (observed)
    "sigma": np.array([1.0, 3.0]),     # arcsec
    "qobs": np.array([0.9, 0.7])
}

# Rest-frame template spectrum setup
lam_spec_template = np.linspace(4000, 7000, 1000)
flux_spec_template = np.exp(-((lam_spec_template - 5500)/200)**2)

# Output wavelength grid (observed-frame)
output_wavelengths = np.linspace(4500, 6500, 500)

# Simulate the data cube
mock = JAMCubeSimulator(
    profile=profile_mge,
    inc=60.,              # Inclination in degrees
    mass_bh=1e8,          # Black hole mass (Msun)
    distance=20.,         # Distance (Mpc)
    band='V',             # Photometric band
    lam=lam_spec_template,
    spectrum=flux_spec_template,
    out_fov=10.,          # FOV side length (arcsec)
    out_pix=0.1,          # Pixel scale (arcsec/pixel)
    out_lam=output_wavelengths,
    redshift=0.05,
    system='AB'
)

# Access the generated 3D data cube (nspec, ny, nx)
cube = mock.cube

```

## References

If you find this software useful for your research, please cite the following papers:

* **Primary Citation:** Nguyen D.D., et al. (2026), MNRAS, 546, stag238 ([ADS](https://ui.adsabs.harvard.edu/abs/2026MNRAS.546ag238N))
* **Jeans Anisotropic Modelling (JAM):** Cappellari M. (2008), MNRAS, 390, 71 ([ADS](https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C))
Cappellari M. (2020), MNRAS, 494, 4819 ([ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C))
* **Multi-Gaussian Expansion (MGE):** Cappellari M. (2002), MNRAS, 333, 400 ([ADS](https://ui.adsabs.harvard.edu/abs/2002MNRAS.333..400C))

## License

Copyright (C) 2026, Michele Cappellari. Distributed under the MIT License.

