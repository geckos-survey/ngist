# nGIST Pipeline Data Products

This document describes the **data products** produced by the nGIST pipeline: file naming, format, structure, and units. It is intended to support **scientific reproducibility** and correct use of the outputs (e.g. in papers or downstream tools). Technical implementation details are kept to a minimum.

**Conventions**

- All product filenames use the run identifier `{RUN_ID}` defined in the pipeline configuration (e.g. a galaxy name). Files are written under the output directory also set in the config.
- **Bin-level** products have **one row (or one spectrum) per Voronoi bin**. **Spaxel-level** products have one value per spaxel; the pipeline also produces **2D maps** (FITS images) built by assigning bin-level or spaxel-level values to spatial coordinates.
- Wavelengths are in **rest frame** unless stated otherwise. The spectral axis is **log λ** (logarithm of wavelength in Å) in binned spectra and best-fit tables; continuum/line cubes use **linear λ** (Å).
- Flux units follow the **input cube**: if the input FITS has a `BUNIT` keyword, that unit is propagated to continuum/line cubes and (where applicable) to HDF5 spectrum metadata. No in-pipeline unit conversion is applied.

---

## 1. Run layout and bookkeeping

| Product       | Format | Description |
|---------------|--------|-------------|
| `CONFIG`      | Text   | Copy of the configuration used for the run (for reproducibility). |
| `LOGFILE`     | Text   | Pipeline log (modules, timings, errors). |

**Reproducibility:** Re-running with the same config file and input data (and same code version) should yield bit-identical results for deterministic modules. Random seeds (e.g. MC realisations) can cause small differences unless fixed.

---

## 2. Spatial preparation

### 2.1 Mask and binning table

| Product           | Format | Structure | Description |
|-------------------|--------|-----------|-------------|
| `{RUN_ID}_mask.fits`   | FITS  | Binary table: one row per spaxel. Column `MASK`: 0 = unmasked, 1 = masked. | Spatial mask (which spaxels are used in binning and fitting). |
| `{RUN_ID}_table.fits`  | FITS  | Multi-extension: primary; table with one row **per spaxel**; WCS/image. Table columns include: `ID` (spaxel index), `BIN_ID` (bin index for this spaxel; ≥0 for binned, &lt;0 for masked with nearest-bin info), `X`, `Y` (spaxel coordinates, e.g. arcsec), `XBIN`, `YBIN` (bin node coordinates), `FLUX`, `SNR`, `SNRBIN`, etc. Header: `PIXSIZE` (e.g. arcsec). | Spaxel–bin mapping and spatial metadata. Used to expand bin-level results to spaxels and to build 2D maps. |

**Reproducibility:** Same mask and target SNR (and covariance settings) give the same Voronoi binning and thus the same `_table.fits` and bin count.

---

## 3. Spectra (HDF5)

Binned and (optionally) full spaxel spectra are stored in **HDF5** for efficiency. Dimensions are **spectral × bins (or spaxels)**.

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_BinSpectra.hdf5`     | HDF5 | Datasets: `SPEC` `(n_wave, n_bins)`, `ESPEC` `(n_wave, n_bins)`; `LOGLAM` `(n_wave,)`. Attributes: `VELSCALE` (km/s per pixel), `CRPIX1`, `CRVAL1`, `CDELT1` (log λ); optional `BUNIT` (flux unit from input). | Log-rebinned spectra **per Voronoi bin** (flux and variance). Same physical units as input. |
| `{RUN_ID}_BinSpectra_linear.hdf5` | HDF5 | Same layout as above; wavelength sampling is **linear** (before log-rebin). | Linear-wavelength binned spectra (intermediate product). |
| `{RUN_ID}_AllSpectra.hdf5`    | HDF5 | Datasets: `SPEC` `(n_wave, n_spaxels)`, `ESPEC` `(n_wave, n_spaxels)`; `LOGLAM` `(n_wave,)`. Same attributes as above. | Log-rebinned spectra **per spaxel** (only written when GAS is run at SPAXEL level). |

**Reproducibility:** Same `VELSCALE` and wavelength range yield the same log λ grid and bin-wise co-addition. Flux units are preserved from the input (BUNIT propagated when present).

---

## 4. Stellar kinematics (KIN)

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_kin.fits` | FITS | Primary + binary table **one row per bin**. Columns: `V`, `SIGMA` (km/s); optional `H3`, `H4`, …; `ERR_*`, `FORM_ERR_*`; optional `REDDENING`; `SNR_POSTFIT`. | Stellar radial velocity and velocity dispersion (and higher moments) per bin. |
| `{RUN_ID}_kin-bestfit.fits` | FITS | Primary; table `BESTFIT` `(n_bins, n_pix)` (best-fit spectrum per bin); `LOGLAM`; `GOODPIX`; table `SPEC` (binned spectra). | Full best-fit stellar spectrum per bin (log λ). |
| `{RUN_ID}_kin-optimalTemplates.fits` | FITS | Tables: optimal templates per bin; `LOGLAM_TEMPLATE`; combined optimal template. | Optimal stellar templates from pPXF weights. |
| `{RUN_ID}_kin-SpectralMask.fits` | FITS | Table: spectral mask per bin (which pixels were used in the fit). | Good-pixel mask per bin. |

**Reproducibility:** Same templates, wavelength mask, and regularization yield the same kinematics and best-fits for a given binned spectrum.

---

## 5. Continuum (CONT) and continuum/line cubes

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_kin-bestfit-cont.fits` | FITS | Primary; table `BESTFIT` `(n_bins, n_pix)`; `LOGLAM`. | Continuum-only best-fit spectrum per bin (log λ). Used to build continuum/line cubes. |
| `{RUN_ID}_CONTcube.fits` | FITS | 3D image `(n_wave, NY, NX)`: continuum flux at **spaxel** resolution (linear λ). Header: WCS, `NAXIS3` = wavelength; optional `BUNIT`, `CUNIT3`. | Continuum-only cube (one value per spaxel per wavelength). Same flux units as input. |
| `{RUN_ID}_LINEcube.fits` | FITS | Same 3D layout. | Emission-only cube (observed − continuum). Same units as input. |
| `{RUN_ID}_ORIGcube.fits` | FITS | Same 3D layout. | Original (observed) cube at same sampling. Same units as input. |

**Reproducibility:** Same CONT fit and same bin–spaxel mapping give the same continuum and line cubes. Units are propagated from the input cube (BUNIT/CUNIT3) when present.

---

## 6. Emission lines (GAS)

Emission-line products are **per bin** (and optionally **per spaxel** when GAS is run at SPAXEL level). Each fitted line has columns for flux, flux error, velocity, velocity error, sigma, sigma error, equivalent width, EW error, and continuum at line centre.

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_gas_BIN.fits` | FITS | Binary table: **one row per bin**. Columns: `BIN_ID`; stellar kinematics (e.g. `V_STARS2`, `SIGMA_STARS2`); per line: `{LINE}_FLUX`, `{LINE}_FLUX_ERR`, `{LINE}_VEL`, `{LINE}_VEL_ERR`, `{LINE}_SIGMA`, `{LINE}_SIGMA_ERR`, `{LINE}_EW`, `{LINE}_EW_ERR`, `{LINE}_CONT`; optional BPT columns. Flux in physical units (e.g. erg/s/cm²); velocities in km/s; EW in Å. | Integrated line fluxes, kinematics, and equivalent widths per bin. **EW sign:** positive = emission, negative = absorption. |
| `{RUN_ID}_gas_SPAXEL.fits` | FITS | Same column layout; **one row per spaxel**. | Same quantities at spaxel resolution (when GAS runs at SPAXEL). |
| `{RUN_ID}_gas-bestfit_{LEVEL}.fits` | FITS | Tables: total best-fit spectrum; gas-only best-fit; `LOGLAM`; good-pixels. | Best-fit spectra (total and gas-only) per bin or per spaxel. |
| `{RUN_ID}_gas-cleaned_{LEVEL}.fits` | FITS | Cleaned (emission-subtracted) spectra. | Spectra after subtracting fitted emission lines. |
| `{RUN_ID}_gas-optimalTemplate_{LEVEL}.fits` | FITS | Optimal gas templates. | Gas templates from the fit. |
| `{RUN_ID}_gas-weights_{LEVEL}.fits` | FITS | Weights for gas components. | Fit weights. |

**Reproducibility:** Same line list, wavelength range, and stellar continuum (KIN/CONT) give the same line fluxes and EWs for a given spectrum. EW convention: **positive = emission**, **negative = absorption**.

---

## 7. Star formation histories (SFH)

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_sfh.fits` | FITS | Binary table: **one row per bin**. Columns include mean age, metallicity, alpha, mass-weighted and light-weighted quantities, and their errors. | Stellar population properties per bin from full-spectral fitting. |
| `{RUN_ID}_sfh-weights.fits` | FITS | Tables: template weights (age, metallicity, alpha); grid info. | Weights of the stellar population templates per bin. |
| `{RUN_ID}_sfh-bestfit.fits` | FITS | Best-fit spectrum per bin; `LOGLAM`. | SFH best-fit spectrum per bin. |
| `{RUN_ID}_sfh-weights-SpectralMask.fits` | FITS | Spectral mask used in SFH fit. | Good-pixel mask. |

**Reproducibility:** Same templates, regularization, and wavelength mask yield the same SFH weights and derived properties for a given binned spectrum.

---

## 8. Line strengths (LS)

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_ls_OrigRes.fits` | FITS | Binary table: **one row per bin**. Columns: line-strength indices (e.g. LIS) at **original** (bin) resolution. | Absorption line-strength indices per bin. |
| `{RUN_ID}_ls_AdapRes.fits` | FITS | Same layout; indices at **adapted** resolution. | Line strengths at adapted spectral resolution. |
| `{RUN_ID}_ls-cleaned_linear.fits` | FITS | Cleaned spectra (linear λ) used for index measurement. | Emission-cleaned spectra. |

**Reproducibility:** Same index definitions and cleaned spectra give the same index values.

---

## 9. 2D maps (FITS)

Maps are **FITS images** with one extension per quantity (e.g. V, SIGMA, or per-line flux). Coordinates come from `_table.fits`; pixels outside the field are NaN.

| Product | Format | Structure | Description |
|---------|--------|-----------|-------------|
| `{RUN_ID}_SPATIAL_BINNING_maps.fits` | FITS | Image extensions: e.g. BINID, FLUX, SNR, SNRBIN, XBIN, YBIN. | Maps of binning outputs. |
| `{RUN_ID}_KIN_maps.fits` | FITS | One extension per kinematics (V, SIGMA, H3, H4, errors, etc.). | Stellar kinematics maps. |
| `{RUN_ID}_GAS_maps.fits` | FITS | One extension per emission line (flux, vel, sigma, EW, etc.) at BIN or SPAXEL level. | Emission-line maps. |
| `{RUN_ID}_SFH_maps.fits` | FITS | One extension per SFH quantity. | SFH maps. |
| `{RUN_ID}_LS_OrigRes_maps.fits` / `_AdapRes_maps.fits` | FITS | One extension per line-strength index. | Line-strength maps. |

**Reproducibility:** Maps are derived deterministically from the bin-level (or spaxel-level) tables and the spaxel–bin mapping in `_table.fits`.

---

## 10. Optional / external products

- **User modules (UMOD):** If enabled, can produce e.g. `{RUN_ID}_twocomp_kin.fits` and corresponding maps; structure depends on the module.
- **IDIA-HDF5 cubes:** If the CONT/LINE/ORIG FITS cubes are converted externally (e.g. with fits2idia) to IDIA-HDF5 format, the result is **external** to the pipeline; units and WCS should match the FITS cubes.
- **PDF maps:** The pipeline can write PDF figures (e.g. kinematics, emission lines, SFH) into a `maps/` subdirectory; filenames follow the run ID and quantity.

---

## 11. Spaxel–bin mapping (reproducibility)

- **Bin-level** FITS tables (kin, sfh, gas_BIN, ls, etc.) have **one row per bin**, with rows ordered by bin index (0, 1, …, n_bins−1).
- **Spaxel–bin correspondence** is given by `_table.fits`: column `BIN_ID` for each spaxel. To assign a bin-level value to spaxels: for each spaxel, read `BIN_ID`; the row index in the bin-level table is the bin index (when BIN_ID ≥ 0).
- **2D maps** are built by filling an image at coordinates `(X,Y)` from `_table.fits` with the value for that spaxel (either from a spaxel-level table or from the bin-level table via `BIN_ID`). No interpolation is performed between bins.

---

## 12. Version and citation

Pipeline version is recorded in the codebase (`_version.py`). For **reproducible** results, record the nGIST version and the configuration file used. Citation: Fraser-McKelvie et al. 2025, A&A 700, 237; nGIST ASCL entry https://ascl.net/2507.015.
