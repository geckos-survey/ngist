# How continuum values are computed and maps are generated

This document describes how continuum (and other bin-level) values are computed and how 2D maps are built. The **artificial structure in the bins** (visible Voronoi tessellation) comes from the fact that all derived quantities are computed **per Voronoi bin**, then **assigned uniformly to every spaxel in that bin** when building maps or cubes—with no smoothing or interpolation between bins.

---

## 1. Data flow: bins vs spaxels

- **Voronoi binning** (`spatialBinning/voronoi.py`) assigns each spaxel to a bin and writes `_table.fits` with one row per spaxel:
  - **ID**: spaxel index (0 … N_spaxels−1).
  - **BIN_ID**: bin index for that spaxel (≥0 for binned spaxels; &lt;0 for masked).
  - **X, Y**: spaxel coordinates (e.g. arcsec).
  - **XBIN, YBIN**: bin node coordinates (replicated to spaxel level).

- **Binned spectra** (`_BinSpectra.hdf5`): one spectrum per **bin** (shape `[n_wave, n_bins]`). All spaxels in a bin share that spectrum.

- **Module results** (KIN, CONT, GAS at BIN level, SFH, LS): one value (or one spectrum) per **bin**. They are stored in tables with **one row per bin** (e.g. `_kin.fits`, `_kin-bestfit-cont.fits`).

So: **values are computed at bin resolution**; they are later expanded to spaxel resolution using the spaxel–bin mapping in `_table.fits`.

---

## 2. How continuum values are computed

### 2.1 Stellar continuum (KIN and CONT modules)

**KIN module** (`stellarKinematics/ppxf_kin_wrapper.py`):

- Input: binned spectra from `_BinSpectra.hdf5` (one spectrum per bin).
- pPXF is run **once per bin** on that bin’s spectrum (optionally with cleaning and MC errors).
- Output per bin: kinematics (V, σ, h3, h4, …) and a **best-fit continuum spectrum** `ppxf_bestfit[bin_index, :]` (log λ sampling).
- Written to:
  - `_kin.fits`: kinematics (one row per bin).
  - `_kin-bestfit.fits`: full best-fit spectrum per bin (BESTFIT HDU: shape `[n_bins, n_pix]`).

**CONT module** (`continuumCube/ppxf_cont_wrapper.py`):

- Input: same `_BinSpectra.hdf5` (again one spectrum per bin).
- pPXF is run **once per bin** (continuum-only fit, same idea as KIN).
- Output: one continuum spectrum per bin, written to **`_kin-bestfit-cont.fits`**:
  - HDU 1 **BESTFIT**: shape `[n_bins, n_pix]` — continuum flux (log λ) for each bin.

So the **continuum is defined only at bin level**: one spectrum per Voronoi bin, no per-spaxel continuum fit.

### 2.2 Continuum cube (CONT → `*_CONTcube.fits`)

`save_maps_fits.saveContLineCube()` builds the continuum (and line) **cube** at spaxel resolution:

1. **Read** `_kin-bestfit-cont.fits`: `ppxf_bestfit` has shape `[n_bins, n_pix]` (one continuum spectrum per bin).
2. **Read** `_table.fits`: `spaxID` = spaxel indices, `binID[s]` = bin index for spaxel `s`.
3. **For each spaxel** `s`:
   - `binID_spax = binID[s]`.
   - If `binID_spax < 0` (masked): continuum set to zero.
   - If `binID_spax >= 0`: take the **bin’s** continuum spectrum:
     - `fitSpec = ppxf_bestfit[binID_spax, :]` (same spectrum for all spaxels in that bin).
   - Interpolate from log λ to linear λ: `CubicSpline(np.exp(logLam), fitSpec)(linLam)` → `fitSpec_lin`.
   - **Per-spaxel scaling**:  
     `obsSignal = median(observed_spectrum[s] in SNR window)`,  
     `fitSignal = median(fitSpec_lin in SNR window)`,  
     then `fitSpec_lin *= obsSignal / fitSignal`.  
     So **shape** of the continuum is identical for all spaxels in a bin; only the **normalization** changes per spaxel to match local observed flux.
4. **Assign**:
   - `contCube[:, s] = fitSpec_lin`,
   - `lineCube[:, s] = obs - fitSpec_lin`,
   - `origCube[:, s] = obs`.

So in the continuum cube:

- **Within a bin**: continuum **shape** is the same (same pPXF fit); **level** varies with the scaling `obsSignal/fitSignal` (so a collapsed “continuum map” can still show gradients within a bin).
- **Across bins**: continuum shape and level change **discontinuously** at Voronoi boundaries, because each bin has its own pPXF fit and no interpolation is done between bins.

That combination (same shape per bin, scaling within bin, step at boundaries) is what can look like “artificial structure” tied to the bins.

---

## 3. How 2D maps are generated (KIN, GAS BIN, SFH, LS)

Map creation in `save_maps_fits.savefitsmaps()` (and the GAS/LS variants) works as follows.

### 3.1 Bin-level modules (KIN, SFH, UMOD)

1. **Read bin-level results** (e.g. `_kin.fits`): table with **one row per bin**; columns are e.g. V, SIGMA, H3, H4, …
2. **Read** `_table.fits`: `binNum_long` = BIN_ID for each spaxel (length = number of spaxels), `ubins` = unique bin IDs.
3. **Expand to spaxel level (“long” version)**:
   - `result_long` has one row per spaxel.
   - For each bin `i`: find all spaxels with `binNum_long == ubins[i]` and set `result_long[idx, :] = result[i, :]`.
   - So **every spaxel in a bin gets exactly the same value** (e.g. same V, same σ) — no interpolation.
4. **Build the 2D image**:
   - Use spaxel coordinates from `_table.fits`: `X`, `Y` (with `X` negated for RA), `pixelsize`.
   - Pixel indices: `i = round((X - xmin) / pixelsize)`, `j = round((Y - ymin) / pixelsize)`.
   - `image[i, j] = val[spaxel_index]` for each spaxel (with index reversal for RA and then transpose for FITS).
   - Pixels that do not correspond to any spaxel stay NaN.

So the map is a **direct stamping of bin-level values onto the spaxel grid**: constant value over each Voronoi bin, sharp steps at bin edges. That is the **artificial structure** — the visible Voronoi pattern.

### 3.2 GAS module (BIN level)

- **Read** `_gas_BIN.fits`: one row per bin (per-line fluxes, EW, etc.).
- **Convert to long**: `idxConvert` from unique bin IDs so `results = results[idxConvert]` gives one value per spaxel (same value for all spaxels in the same bin).
- **Image**: same as above — fill image at `(X,Y)` with that spaxel’s value.

Again: **one value per bin** → same value for every spaxel in the bin → **Voronoi pattern** in the map.

### 3.3 GAS SPAXEL and line/continuum cubes

- **GAS SPAXEL**: fits are already per spaxel, so no “bin → spaxel” expansion; map is built directly from spaxel-level values (no bin stepping).
- **Continuum cube**: as in §2.2, values are assigned per spaxel (bin continuum + per-spaxel scaling). Collapsing that cube to a map (e.g. mean over λ) gives a map that can vary within a bin (due to scaling) but still has **discontinuities at bin boundaries** because the underlying continuum shape changes only at bin edges.

---

## 4. Summary: why continuum maps show artificial bin structure

1. **Continuum is computed per Voronoi bin** (one pPXF fit per bin in KIN/CONT). There is no per-spaxel continuum fit.
2. **Assignment to spaxels**:
   - In the **continuum cube**: each spaxel gets its bin’s continuum spectrum, then a **per-spaxel normalization** (obsSignal/fitSignal). So within a bin the continuum shape is identical; only the level changes. At bin boundaries the **shape** (and often level) jumps.
   - In **bin-level maps** (KIN, GAS BIN, SFH, LS): each spaxel gets exactly the **same** scalar value as every other spaxel in its bin.
3. **No smoothing**: Maps and cubes use the spaxel–bin mapping as-is. There is no interpolation or smoothing between bins, so **Voronoi edges appear as sharp boundaries** in any continuum-derived map or in kinematics/maps from bin-level modules.

So the “artificial structure in the bins” is the **direct visualization of the Voronoi tessellation**: values are constant (or same-shaped) within each bin and change discontinuously at bin boundaries by design.

---

## 5. Shell-like structure (continuum and EW only)

**Symptom:** Continuum-derived products (e.g. continuum cube collapsed to a map, or EW maps that use the continuum) can show **shell-like** or **concentric** structure that does **not** appear in KIN maps or other GAS maps (flux, velocity, etc.).

**Cause:** In `saveContLineCube`, the continuum was originally scaled **per spaxel**: `fitSpec_lin *= obsSignal(s) / fitSignal(bin)`. Here `obsSignal(s)` is the observed median flux in the SNR window for **spaxel s** (varies smoothly with position), and `fitSignal(bin)` is the median of the **bin's** fit spectrum (same for all spaxels in the bin). So the continuum level at each spaxel was (bin continuum) × (obsSignal(s) / fitSignal(bin)). Within a bin the level varied with obsSignal(s), while the factor 1/fitSignal(bin) **jumped at bin boundaries**. If fitSignal correlates with radius (e.g. inner vs outer bins), that produced **concentric bands** (shells). KIN and most GAS maps use a **single value per bin** (no per-spaxel scaling), so they do not show this effect; EW uses the continuum in the denominator, so it inherited the same shell pattern.

**Fix:** Continuum cube construction now uses **per-bin scaling** instead of per-spaxel: for each bin, one scale factor is computed (median of obsSignal over all spaxels in that bin / fitSignal for that bin) and applied to the bin's continuum for **all** spaxels in the bin. The continuum is then **constant within each bin** at a given wavelength (consistent with KIN/GAS), and the shell-like artifact is removed.
