# ngistPipeline Memory Management Plan

## Context

Memory usage increased on the same data after recent dev-mauve changes (Jan 29–30). Resource logs show two major RAM spikes: one when data are read and one when the continuum cube is written. This document identifies the sources and proposes fixes.

## Recent commits affecting memory (dev-mauve)

| Commit   | Change |
|----------|--------|
| `5a7377e` | FITS `memmap=True` and context managers for voronoi, kin-guess, save_maps_fits |
| `7f09a19` | Vectorized DER_SNR (`der_snr_2d`) when variance extension is missing |
| `c949ed9` | Vectorized spatial binning and GAS continuum expansion |
| `7154145` | Vectorized optimal template (`templates @ weights`) in KIN, CONT, SFH |

The vectorized DER_SNR and other changes can increase **peak** memory because they operate on full arrays instead of one spaxel/column at a time, even though they are faster.

---

## Spike 1: Data read (readData / MUSE_WFM.readCube)

### Where

`ngistPipeline/readData/MUSE_WFM.py` (and equivalent MUSE_*.py readers).

### Causes

1. **FITS data is copied when reshaped**
   - `data = hdu[ihdu].data` (memmapped) then `spec = np.reshape(data, [s[0], s[1]*s[2]])`.
   - Reshaping (nwave, ny, nx) → (nwave, ny*nx) typically creates a **copy** when the layout is not C-contiguous in the target shape, so a full in-memory copy of the cube is made for `spec`.

2. **Variance path**
   - With variance extension: `stat = hdu[2].data` and `espec = np.reshape(stat, ...)` → again a full copy (or view), then `espec = espec[idx, :]` → another copy when trimming wavelength.
   - Without variance (DER_SNR path): `der_snr_2d(spec)` allocates:
     - `flux = np.where(flux == 0.0, np.nan, flux)` → **full copy** of `spec`;
     - `diff = 2*flux[2:-2,:] - flux[:-4,:] - flux[4:,:]` → **large temporary** (nwave-4 × nspaxels);
     - `espec = np.broadcast_to(noise_per_spaxel.reshape(1,-1), spec.shape).copy()` → **full copy** for `espec`.
   So the vectorized DER_SNR path holds **spec + flux copy + diff + espec** at once, increasing peak memory vs the old per-spaxel loop (which only had `spec` + `espec` filled incrementally).

3. **Wavelength trimming**
   - `spec = spec[idx, :]` and `espec = espec[idx, :]` use fancy indexing → **copies**. Until these run, the full untrimmed arrays are in memory.

4. **Extinction correction**
   - `spec = spec / reshaped_extinction_curve` (and same for `espec`) creates new arrays unless done in-place.

### Recommendations (read path)

| Priority | Action |
|----------|--------|
| **High** | **Avoid full copies in DER_SNR path**: In `der_snr_2d`, avoid a full-sized copy for the zero→NaN mask. Options: (a) use a read-only view and pass `where=(flux != 0)` into nanmedian where possible, or (b) process in chunks of spaxels so only one chunk of `flux`/`diff` is allocated at a time (reduces peak at cost of more passes). |
| **High** | **Trim wavelength early**: After reading header and shape, compute wavelength and `idx` for LMIN_TOT/LMAX_TOT. Read only the needed slice from the FITS data (e.g. `data[idx, :, :]`) so that `spec`/`espec` are never full-cube size in memory. This may require refactoring the reader to slice before reshape. |
| **Medium** | **Reshape without copy where possible**: Use `np.reshape(..., order='C')` and ensure we don’t force a copy; if the FITS cube is (nwave, ny, nx), reshaping to (nwave, ny*nx) can be a view. Verify with a small test and prefer in-place or view where safe. |
| **Medium** | **In-place extinction**: Use `np.divide(spec, curve, out=spec)` and same for `espec` to avoid extra allocations. |
| **Low** | **Release references after trim**: Explicitly `del` large temporaries (e.g. after `espec = espec[idx, :]`) so the GC can reclaim sooner (optional; often redundant if names go out of scope). |

---

## Spike 2: Continuum cube writing (saveContLineCube)

### Where

`ngistPipeline/writeFITS/save_maps_fits.py` → `saveContLineCube(config)`.

### Causes

1. **Redundant full cube read**
   - Line 348: `inputCube = readCube(config)` reads the **entire MUSE cube again** from disk. The main pipeline already read the cube once and dropped it (`del cube` in MainPipeline after prepareSpectra). So we temporarily hold a **second full copy** of the cube only to build CONT/LINE/ORIG cubes.

2. **Three full-size cubes at once**
   - Lines 372–374: `contCube`, `lineCube`, and `origCube` are each `np.full([len(linLam), NY*NX], np.nan)`. So we have **three** full (nwave × nspaxels) arrays in memory simultaneously.

3. **Per-bin arrays**
   - `fitSpec_lin_per_bin = np.zeros((n_bins, len(linLam)))` and the loop that fills it (lines 388–402) add more memory but are small compared to the cubes.

4. **Bestfit table**
   - `ppxf_bestfit = np.array(cont_hdu[1].data.BESTFIT)` loads the full bestfit table; this is required but contributes to peak.

So peak memory in `saveContLineCube` ≈ **one full input cube (from readCube) + ppxf_bestfit + fitSpec_lin_per_bin + contCube + lineCube + origCube** ≈ 4× cube size plus smaller terms.

### Recommendations (saveContLineCube)

| Priority | Action |
|----------|--------|
| **High** | **Avoid second full cube read**: Do not call `readCube(config)` inside `saveContLineCube`. Instead, either: (A) **Stream from FITS**: open the input cube with `memmap=True` and only read the wavelength slice needed for CONT (LMIN–LMAX), e.g. one slice or chunk at a time when filling `origCube`/spectra; or (B) **Reuse binned spectra**: CONT already has binned spectra in HDF5 and bestfit per bin; we need to expand bin→spaxel and need the **original (unbinned) spaxel spectra** only for the LINE/ORIG cubes. For that, read the input cube in **chunks** (e.g. by wavelength or by spatial regions) and fill `spectra_all` (or fill `origCube`/`lineCube` directly) chunk by chunk, then delete the chunk. Prefer one shared path that reads the input cube once in a streaming/chunked way. |
| **High** | **Write one cube at a time**: Instead of allocating `contCube`, `lineCube`, and `origCube` together, compute and write them one by one: e.g. (1) build and write CONT cube, then `del contCube`; (2) build and write LINE cube (using spectra and continuum, or recompute from same chunked read), then `del lineCube`; (3) build and write ORIG cube (or stream it from the same chunked read). This reduces peak from 3× to ~1× cube for the output buffers. |
| **Medium** | **Chunked read of input**: When reading the MUSE cube for CONT/LINE/ORIG, use a single `fits.open(..., memmap=True)` and slice `data[idx_lam, :, :]` (or equivalent) so that we never materialize the full cube in one go; iterate over y/x or wavelength chunks and fill the three cubes (or each cube in turn) from the memmapped array. |
| **Low** | **Float32 for output cubes**: Already using `np.float32` for the written cubes; ensure we don’t create float64 intermediates when float32 is enough. |

---

## Implementation order

1. **saveContLineCube** (bigger win): Remove redundant `readCube`, introduce chunked/streaming read of the input cube and write CONT/LINE/ORIG one cube at a time.
2. **Read path**: Wavelength trim early (slice before reshape); DER_SNR path reduce temporaries (chunked or in-place where safe); in-place extinction.
3. **Verification**: Re-run the same dataset with psrecord/memory_profiler and compare peak RSS before/after.

---

## Implemented (this session)

- **Read path (MUSE_WFM)**: (1) Wavelength trim early: read only `data[idx, :, :]` (LMIN_TOT..LMAX_TOT) so the full cube is never in memory; (2) in-place extinction with `np.divide(..., out=spec)`; (3) shape from header (`NAXIS1/2/3`) so the first open only reads header.
- **DER_SNR**: `der_snr_2d` now uses `np.ma.masked_equal(flux, 0.0)` instead of `np.where(..., np.nan, flux)` to avoid a full copy of the input for zero-masking.
- **saveContLineCube**: (1) Replaced `readCube(config)` with `_load_input_spectra_trimmed(config)` that opens the input FITS with memmap and loads only the wavelength slice (LMIN_TOT..LMAX_TOT), with redshift and extinction; (2) write CONT cube first and `del contCube`, then build and write LINE, `del lineCube`, then build and write ORIG so only one output cube is in memory at a time.

---

## Summary table

| Location              | Cause of spike                         | Main fix |
|-----------------------|----------------------------------------|----------|
| MUSE_WFM.readCube     | Reshape copy; DER_SNR full-array copies; late trim | Trim early; chunked/smaller DER_SNR temporaries; in-place ops |
| saveContLineCube      | Second full readCube + 3 full cubes    | Chunked/streaming read; write one cube at a time |

This plan should be updated as changes are implemented and re-profiled.

---

## HDF5 vs FITS cube size (why HDF5 products are 3–5× larger)

**Is repeat running appending rather than overwriting?** No. The pipeline does **not** append to HDF5 on re-runs:

- **prepareSpectra** opens HDF5 with `h5py.File(..., 'w')` (write/overwrite). There is no `'a'` (append) mode used for spectrum files.
- When `OW_OUTPUT` is `False`, the prepareSpectra **module** is skipped entirely if all HDF5 files already exist (`_BinSpectra.hdf5`, `_BinSpectra_linear.hdf5`, and optionally `_AllSpectra.hdf5`). So no write, no append.

**Why are HDF5 spectrum files 3–5× the size of comparable FITS cube files?**

1. **Two arrays per file**  
   Each HDF5 spectrum file stores **SPEC** (flux) and **ESPEC** (variance/error) of the same shape. A single FITS cube (e.g. `_CONTcube.fits`) stores one 3D array. So per “logical” cube, HDF5 holds ~2× the data (flux + error).

2. **Dtype**  
   prepareSpectra uses `log_spec.dtype` / `log_error.dtype` (from ppxf `log_rebin`), which are typically **float64**. The continuum/line/orig FITS cubes in `save_maps_fits` are written as **float32**. So 8 bytes vs 4 bytes per value → **2×** from dtype alone.

3. **No compression**  
   HDF5 datasets in prepareSpectra are created without compression (`create_dataset(..., )` has no `compression=`). FITS can be written with compression (e.g. Rice) in some workflows, further reducing FITS size.

Combined: **2 (flux+error) × 2 (float64 vs float32) ≈ 4×** raw data, plus HDF5 chunk/B-tree/metadata overhead and no compression, readily explains **3–5×** larger HDF5 files than a single FITS cube.

**Optional future reduction:** Use `dtype=np.float32` and/or `compression="gzip"` (or `"lzf"`) when creating SPEC/ESPEC in prepareSpectra if float32 precision and smaller files are preferred.
