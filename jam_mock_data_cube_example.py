# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %% [markdown]
# # Usage Example for jam_mock_data_cube
#
# Michele Cappellari, Oxford, 12 June 2024

# %%
from urllib import request
from pathlib import Path
from time import perf_counter as clock

import matplotlib.pyplot as plt
import numpy as np

from plotbin.display_pixels import display_pixels
from plotbin.plot_velfield import plot_velfield
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

from jam_mock_data_cube import JAMCubeSimulator

# %% [markdown]
# ## Parameters of the MGE model

# %%
surf, sigma, qobs = np.loadtxt('NGC300_hst_08599_02_wfpc2_f814w_pc_sci_lumdensity_cuspGaussian.txt').T

# %% [markdown]
# ## Parameters of the JAM model

# %%
distance = 16.5     # Assume Virgo distance in Mpc (Mei et al. 2007)
mbh = 1e8           # Black hole mass in solar masses
beta = np.full_like(surf, 0)
inc = 60    # deg

# %% [markdown]
# ## Parameters of the instrument

# %%
out_fov = .2    # arcsec FoV
out_pix = .01/3   # 10 mas scale oversampled by 3x
out_lam_range = [14e3, 18e3]    # Angstroms
out_dlam = 0.4/3                # Angstroms oversampled by 3x
redshift = 0
band_mge = 'hst/wfpc2_f814w'

# %% [markdown]
# ## Input mock spectrum
#
# Read SPS models file from my GitHub if not already in the `pPXF` package dir.
# The SPS model files are also available on my [GitHub Page](https://github.com/micappe/ppxf_data).

# %%
sps_name = 'xsl'        # Use X-Shooter spectral library
ppxf_dir = Path(util.__file__).parent
basename = f"spectra_{sps_name}_9.0.npz"
filename = ppxf_dir / 'sps_models' / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)
a = np.load(filename)
templates, ages, metals, lam = a["templates"], a["ages"], a["metals"], a["lam"]
spectrum = templates[:, -2, -2]  # age = 12.59 Gyr [M/H] = 0

# %% [markdown]
# Run the simulation

# %%
mock = JAMCubeSimulator(surf, sigma, qobs, inc, mbh, distance, band_mge, lam, spectrum,
                        out_fov, out_pix, out_lam_range, out_dlam, redshift, align='cyl', beta=beta)

# %% [markdown]
# Show the output surface brightness

# %%
display_pixels(mock.x, mock.y, mock.spectra.sum(0))

# %% [markdown]
# ## Test recovery of input kinematics with pPXF

# %%
velscale = 30  # km/s
galaxy_cube, ln_lam = util.log_rebin(mock.lam, mock.spectra, velscale)[:2]
template, ln_lam_temp = util.log_rebin(lam, spectrum, velscale)[:2]

template /= np.median(template)
galaxy_cube /= np.median(galaxy_cube)
goodpixels = np.arange(100, len(galaxy_cube) - 100)
velbin, sigbin = np.empty((2, galaxy_cube.shape[1]))

# %%
t = clock()
for j, galaxy in enumerate(galaxy_cube.T):
    pp = ppxf(template, galaxy, np.ones_like(galaxy), velscale, [0, 200], quiet=1, degree=-1,
              lam=np.exp(ln_lam), lam_temp=np.exp(ln_lam_temp), goodpixels=goodpixels)
    velbin[j], sigbin[j] = pp.sol
    if not j % 200:
        print(f"{j}/{mock.x.size}")
print(f"Elapsed time {clock() - t:.2f} s")

# %%
plt.clf()
plt.subplot(221)
vmin, vmax = np.min(velbin), np.max(velbin)
plt.title("pPXF output $V$")
plot_velfield(mock.x, mock.y, velbin, colorbar=1, flux=mock.surf_band, 
              vmin=vmin, vmax=vmax, nodots=1)

plt.subplot(222)
plt.title("JAM input $V$")
plot_velfield(mock.x, mock.y, mock.vel, colorbar=1, flux=mock.surf_band, 
              vmin=vmin, vmax=vmax, nodots=1)

plt.subplot(223)
vmin, vmax = np.min(sigbin), np.max(sigbin)
plt.title("pPXF output $\\sigma$")
plot_velfield(mock.x, mock.y, sigbin, colorbar=1, flux=mock.surf_band, 
              vmin=vmin, vmax=vmax, nodots=1) 

plt.subplot(224)
plt.title("JAM input $\\sigma$")
plot_velfield(mock.x, mock.y, mock.sig, colorbar=1, flux=mock.surf_band, 
              vmin=vmin, vmax=vmax, nodots=1)
plt.tight_layout()

# %% [markdown]
# Test calibration (see Fig.1 in 
# [Neumayer et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26ARv..28....4N))
# Our magnitude at `R=0.1` arcsec seems about 1 mag fainter than theirs...

# %%
plt.plot(np.sqrt(mock.x**2 + mock.y**2), mock.mag_mge, 'o')
plt.xlabel("Radius (arcsec)")
plt.ylabel("Surface Brightness (mag arcsec$^{-2}$)")
plt.xscale('log') 
plt.title(f"Surface Brightness {band_mge}")
plt.gca().invert_yaxis()
