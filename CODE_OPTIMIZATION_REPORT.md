im# nGIST Pipeline — Performance and Memory Optimization Report

This report summarizes findings from a review of the ngistPipeline codebase with a focus on **performance** and **memory** usage, and provides concrete recommendations.

---

## 1. Executive Summary

- **Data I/O**: Main cube reads use `memmap=True` (good); several FITS opens do not; DER_SNR runs in a per-spaxel Python loop when variance is missing.
- **Binning & prepareSpectra**: Co-addition is a Python loop over bins; log-rebinning uses chunked I/O (good) but runs spectra and errors in two separate passes; spatial binning loop could be vectorized or parallelized.
- **pPXF modules (KIN, CONT, GAS, SFH)**: Joblib parallel + memmap is used for KIN/CONT; inner MC loops and optimal-template construction use Python loops; some FITS opens lack `memmap`; CONT has a stray `import code`.
- **writeFITS / maps**: Multiple FITS files opened without `memmap`; map construction loops over extensions one-by-one (acceptable but could batch).

Recommendations are ordered by impact: high (clear win), medium (worth doing), low (nice to have).

---

## 2. Data Loading and I/O

### 2.1 Read Data (e.g. `readData/MUSE_WFM.py`)

| Finding | Severity | Recommendation |
|--------|----------|-----------------|
| Main cube uses `fits.open(..., memmap=True, lazy_load_hdus=True)` | Good | Keep as is. |
| When variance extension is missing, variance is estimated with `der_snr` in a **Python loop** over all spaxels: `for i in range(0, spec.shape[1]): espec[:, i] = der_snr.der_snr(spec[:, i])` | **High** | **Vectorize or parallelize:** (1) Add a vectorized DER_SNR that operates on a 2D array (e.g. using `numpy` slicing for the 2*flux_i - flux_i-2 - flux_i+2 term), or (2) use `joblib.Parallel` over chunks of spaxels so many spectra are estimated in parallel. |
| Same per-spaxel DER_SNR loop appears in MUSE_NFM, MUSE_NFMAO, MUSE_WFMAON, MUSE_WFMAOE | **High** | Apply the same vectorization/parallelization in all readers that use DER_SNR. |

### 2.2 FITS usage across the codebase

| Location | Issue | Recommendation |
|----------|--------|-----------------|
| `spatialBinning/voronoi.py` | `fits.open(maskfile)` with no `memmap=True` | Use `fits.open(maskfile, memmap=True)` so the mask is not fully loaded when only a small column is needed. |
| `stellarKinematics/ppxf_kin_wrapper.py`, `continuumCube/ppxf_cont_wrapper.py` | `fits.open(... "_kin-guess.fits")` without memmap | Use `memmap=True` when only reading guess columns. |
| `writeFITS/save_maps_fits.py` | Multiple `fits.open(...)` without memmap for table/map FITS | Use `memmap=True` (and `with` context) where only parts of the file are read. |
| `prepareSpectra/default.py` | Mask and table already use `memmap=True` / `mem_map=True` | Keep; consider standardising keyword to `memmap=True` (astropy standard). |

Using `memmap=True` consistently for read-only FITS access reduces peak RAM and speeds up startup when only subsets of files are needed.

### 2.3 HDF5 usage

- **prepareSpectra**: `run_log_rebinning` uses a chunk size (e.g. 1000) for processing and writing; chunked HDF5 writes in `saveAllSpectra` and `saveBinSpectra` are in place. **No change required** for memory.
- **KIN/CONT/GAS/SFH**: BinSpectra is read with `h5py.File(..., 'r')` and full datasets loaded with `f['SPEC'][:][idx_lam, :]`. For very large cubes, consider reading only required wavelength chunks (e.g. `f['SPEC'][idx_lam, :]` if layout allows) to avoid loading the full spectral axis.

---

## 3. Binning and prepareSpectra

### 3.1 Spatial co-addition (`prepareSpectra/default.py` — `spatialBinning()`)

- Spectra in the same Voronoi bin are co-added in a **Python loop** over bins: `for i in range(nbins): ... k = np.where(binNum == ubins[i])[0] ...`.
- For large numbers of bins this can be slow and does not use multiple cores.

**Recommendations:**

- **Medium:** Precompute for each bin the list of spaxel indices (or a sparse matrix) once, then use vectorized numpy operations (e.g. `np.add.at` or binned sum via indexing) to compute `bin_data` and `bin_error` without a Python loop over bins.
- **Low:** If keeping the loop, parallelize over chunks of bins with `joblib.Parallel` (same pattern as in ppxf wrappers).

### 3.2 Log-rebinning (`prepareSpectra/default.py`)

- **Good:** Chunked processing in `run_log_rebinning` (e.g. `chunk_size=1000`) avoids building a huge array in memory.
- Spectra and errors are log-rebinned in **two separate** calls to `run_log_rebinning` (and two passes over the cube). 

**Recommendation (low):** Consider a single function that log-rebins both `spec` and `error` in one pass over chunks (same chunk indices), to improve cache use and possibly reduce Python overhead; keep the same chunk size for memory safety.

---

## 4. pPXF Loops and Parallelism (KIN, CONT, GAS, SFH)

### 4.1 Thread control (`MainPipeline.py`)

- `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OMP_NUM_THREADS` are set to 1. This avoids oversubscription when using `joblib.Parallel` and is **good practice**. Keep as is.

### 4.2 Stellar kinematics and continuum (KIN / CONT)

- **Good:** When `PARALLEL=True`, joblib is used with chunked bins, and large arrays (`templates`, `bin_data`, `noise`) are dumped to disk and loaded with `mmap_mode='r'` so workers share memory-mapped data.
- **Optimal template:** In `run_ppxf`, the optimal template is built with a Python loop: `for j in range(0, templates.shape[1]): optimal_template += templates[:, j] * normalized_weights[j]`. This is equivalent to a single matrix–vector product.

**Recommendation (medium):** Replace that loop with `optimal_template = templates @ normalized_weights` (or `np.dot(templates.T, normalized_weights)`). Same change in `continuumCube/ppxf_cont_wrapper.py` and `starFormationHistories/ppxf_sfh_wrapper.py` where the same pattern appears.

- **MC simulations:** Inside `run_ppxf`, MC iterations run in a Python loop (`for o in range(0, nsims)`). Parallelising this inside each bin would complicate the current design; a **low**-priority option is to run MC in parallel per bin (e.g. each worker does one bin including its nsims MC runs) so that MC work is spread across cores.
- **CONT:** Remove the unused `import code` at the top of `continuumCube/ppxf_cont_wrapper.py`.

### 4.3 Emission lines (GAS)

- ppxf_gas_wrapper uses a per-bin (or per-spaxel) loop; parallelisation pattern is similar to KIN/CONT where applicable. KIN bestfit is loaded with `mem_map=True` in `_load_kin_stellar_continuum`; table and FITS opens elsewhere should be checked for `memmap=True` where read-only.
- Stellar continuum expansion from bin to spaxels uses a Python loop over spaxels; this could be vectorized with integer indexing: `stellar_cont_spaxel = stellar_cont_bin[bin_id, :]` (with care for negative bin IDs).

### 4.4 Star-formation histories (SFH)

- Same optimal-template loop as above; replace with a single dot product.
- Same pattern: parallel over bins with joblib, optional memmap for large arrays. FITS/HDF5 opens already use `mem_map=True` / context managers in several places; apply consistently.

---

## 5. writeFITS and Map Generation

### 5.1 `writeFITS/save_maps_fits.py`

- **savefitsmaps:** Opens `_table.fits` and, depending on module, `_kin.fits`, `_sfh.fits`, etc., without `memmap=True`. For read-only use, pass `memmap=True` and use `with fits.open(...) as ...` where possible so files are closed and memory-mapped where beneficial.
- Map building loops over result columns/extensions one by one to build image HDUs; this is acceptable. If many maps are written, batching header creation could slightly reduce overhead (low priority).
- **savefitsmaps_GASmodule:** Same suggestion: use `fits.open(..., memmap=True)` and context managers for table and result FITS.

### 5.2 Redundant FITS opens

- In several modules, the same FITS file is opened more than once (e.g. table and then kin/sfh). Where the same script opens the same file in multiple places, consider opening once and passing the HDU list or required data (medium priority, reduces I/O and descriptor usage).

---

## 6. Other Notes

### 6.1 DER_SNR return value (`readData/der_snr.py`)

- The docstring says the function returns SNR (signal/noise), but the implementation returns `noise` (and the line `return float(signal / noise)` is commented out). Call sites (e.g. MUSE_WFM) assign this to `espec`, i.e. they use it as a noise (or variance) value. If the intended product is variance, the naming/docstring should be clarified; if SNR was intended elsewhere, this may be a bug. **Recommendation:** Clarify or fix the return value and document whether callers expect noise or SNR.

### 6.2 Duplicate code

- `robust_sigma` (and similar helpers) are duplicated across `stellarKinematics/ppxf_kin_wrapper.py`, `continuumCube/ppxf_cont_wrapper.py`, and `starFormationHistories/ppxf_sfh_wrapper.py`. **Recommendation (low):** Move to a single helper (e.g. `ngistPipeline/auxiliary`) and import it to reduce drift and ease maintenance.

---

## 7. Priority Summary

| Priority | Area | Action |
|----------|------|--------|
| **High** | readData (MUSE_WFM, etc.) | Vectorize or parallelize DER_SNR when variance is missing; avoid per-spaxel Python loop. |
| **High** | FITS I/O | Use `memmap=True` (and `with`) in voronoi.py, kin/cont guess files, writeFITS/save_maps_fits, and other read-only FITS opens. |
| **Medium** | pPXF wrappers | Build optimal template with `templates @ normalized_weights` in KIN, CONT, SFH. |
| **Medium** | prepareSpectra | Vectorize or parallelize spatial co-addition (binned sum) in `spatialBinning()`. |
| **Medium** | FITS | Reduce duplicate opens of the same file; use one open and pass data. |
| **Low** | prepareSpectra | Single-pass log-rebinning for spec and error in one chunked pass. |
| **Low** | CONT | Remove `import code`. |
| **Low** | GAS | Vectorize bin→spaxel continuum expansion with indexing. |
| **Low** | Shared code | Centralise `robust_sigma` and document DER_SNR return value. |

---

*Report generated from review of ngistPipeline codebase (data loading, binning, prepareSpectra, stellarKinematics, continuumCube, emissionLines, starFormationHistories, writeFITS).*
