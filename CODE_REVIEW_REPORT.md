# ngistPipeline Code Review Report

**Scope:** Scientific validity, bugs, and consistency of docstrings/comments across the ngistPipeline codebase.  
**Date:** 2025-01-29.  
**Reviewer:** Automated code review (rigorous pass).

---

## Executive summary

The pipeline is scientifically sound in its core design (pPXF-based kinematics/continuum/emission, Voronoi binning, EW convention). Several **bugs** (including one that crashes serial CONT runs), **docstring/comment inaccuracies**, and **robustness gaps** were identified. Recommendations are ordered by severity.

---

## 1. Bugs (must fix)

### 1.1 CONT module: NameError in serial mode (continuumCube/ppxf_cont_wrapper.py)

**Severity: Critical — pipeline crash**

When `config["GENERAL"]["PARALLEL"]` is `False`, the code still executes:

```python
# Remove the memory-mapped files
os.remove(templates_filename_memmap)
os.remove(bin_data_filename_memmap)
os.remove(noise_filename_memmap)
```

These variables are only defined inside the `if config["GENERAL"]["PARALLEL"] == True:` block. In serial mode they are never set, so **NameError** is raised and the CONT step fails.

**Fix:** Move the three `os.remove(...)` lines into the `if config["GENERAL"]["PARALLEL"] == True:` block (immediately after `printStatus.updateDone("Running PPXF in parallel mode", ...)`), so cleanup runs only when memmap files were created. Do not run these removals in the serial branch.

---

### 1.2 addPathsToConfig: possible IndexError (initialise/_initialise.py)

**Severity: High**

In `addPathsToConfig`, each non-comment line is split with `line = line.split('=')` and then `line[1]` is used. If a line contains no `=`, `line` has length 1 and `line[1]` raises **IndexError** (e.g. blank lines or malformed entries in the defaultDir file).

**Fix:** After `line = [x.strip() for x in line]`, add:

```python
if len(line) < 2:
    continue
```

so that only valid `key=value` lines are processed.

---

### 1.3 getLSF: UMOD overwrites LS (auxiliary/_auxiliary.py)

**Severity: Medium**

LSF template file is set with a chain of `if/elif` for `module_used in ("KIN", "CONT", "GAS", "SFH", "LS")`, but the UMOD case uses a separate `if module_used == "UMOD":`, so it is not part of the chain. For `module_used == "LS"`, `lsfTempFile` is set correctly, but then the UMOD `if` is evaluated; for UMOD it would overwrite, but for LS it does not. So the logic bug is: **UMOD should be `elif`**, otherwise the control flow is confusing and any future change could make LS get overwritten by UMOD if both were to be considered. As written, only UMOD is wrong when another module was intended.

**Fix:** Change `if module_used == "UMOD":` to `elif module_used == "UMOD":` so all modules are in one if/elif chain and UMOD does not overwrite a previous choice.

---

## 2. Scientific validity and robustness

### 2.1 Equivalent width (EW)

- **Convention:** EW is implemented as `EW = -F_line / f_cont` (negative for emission). This matches the stated “standard astronomical convention” and the docstrings in `ppxf_gas_wrapper.py` and `gandalf_gas_wrapper.py`. **No change needed.**
- **Continuum at line:** Uses linear interpolation in wavelength for `f_cont` at the line center; division-by-zero and invalid values are handled with `np.where` and `np.nan`. **Reasonable.**
- **EW error:** Error propagation uses `sigma_EW ≈ |EW| * sqrt((sigma_F/F)^2 + (sigma_cont/f_cont)^2)`. Continuum uncertainty is approximated by the local scatter of the stellar continuum in a 10-pixel window. This is a **simplification** (no full covariance); acceptable for pipeline use but should be understood when quoting EW errors.
- **Out-of-range lines:** If a line’s rest wavelength is outside the spectrum (e.g. `lam_line < wave[0]`), `np.searchsorted` and `np.clip` still yield valid indices, but the interpolation can **extrapolate** (e.g. negative `frac`). EW then uses an extrapolated continuum, which can be unreliable. Consider clipping `frac` to `[0, 1]` or setting EW to `np.nan` when the line is outside the wavelength range.

### 2.2 KIN stellar continuum for EW (emissionLines)

- `_load_kin_stellar_continuum` uses `_kin-bestfit.fits` (KIN module output), not the continuum cube. For SPAXEL level it checks `n_kin_bins != np.max(bin_id) + 1`. This **assumes BIN_ID is 0-based and contiguous**. If Voronoi bin IDs were non-contiguous, the check and indexing could be wrong. Currently Voronoi outputs 0, 1, …, n_bins−1, so this is fine; document the assumption or use `len(np.unique(bin_id[bin_id >= 0]))` for robustness.

### 2.3 Continuum cube (save_maps_fits.saveContLineCube)

- Per-bin scaling and `bin_id_to_idx` mapping are correctly used so that continuum is constant within each bin and matches the CONT module’s bin order. **Scientifically consistent.**
- If `len(ubins) > ppxf_bestfit.shape[0]` (e.g. table from another run), the loop breaks and some bins get no fit; `scale_per_bin` can then be 0 or NaN and those bins get zero or constant continuum. Consider a check `len(ubins) <= ppxf_bestfit.shape[0]` and a clear warning or error when violated.

### 2.4 LSF and redshift (auxiliary getLSF)

- LSF wavelengths are divided by `(1 + redshift)` to rest frame; the same is applied when capping the LSF at 2.51 Å. **Physically correct** for rest-frame fitting.

---

## 3. Docstrings and comments (inconsistencies / inaccuracies)

### 3.1 Wrong module attribution

- **writeFITS/save_maps_fits.py** (around 437–446): Comment says “get PPXF best fit continuum from **kinematics** module” and “get logLam from best fit (**continuum/kinematics**) module”. The file read is `_kin-bestfit-cont.fits`, which is produced by the **CONT (continuum)** module, not the KIN module. KIN writes `_kin-bestfit.fits` (full best fit, not continuum-only).
- **Fix:** Replace with “continuum (CONT) module” (or “CONT module”) in both comments.

### 3.2 readMasterConfig (initialise/_initialise.py)

- Docstring: “Read the MasterConfig file and stores all parameters in the configs dictionary.”
- The function accepts `galindex` but **does not use it** (single-galaxy YAML config). Either use `galindex` (e.g. for multi-galaxy configs) or drop it and update the docstring/call sites.

### 3.3 MainPipeline.py

- `skipGalaxy` prints “The **GIST** pipeline” while the success message uses “**nGIST** pipeline”. Unify to “nGIST” for consistency.
- `numberOfGalaxies` is **never used**; `ngalaxies` is hardcoded to 1. Comment in code references “Amelia changed this…”. Either remove the function or use it and document the single-galaxy assumption.

### 3.4 addPathsToConfig (initialise/_initialise.py)

- Inline comment typo: “**Amrlia**” → “Amelia”.
- The branch `elif line[1] == "outputDir" and line[0] == "configDir"` handles the case where the defaultDir file sets `configDir=outputDir` (i.e. config dir = output dir). Logic is correct but depends on `len(line) >= 2`; see Bug 1.2.

### 3.5 getLSF (auxiliary/_auxiliary.py)

- Docstring lists “module = 'KIN', 'CONT', 'GAS', 'SFH', or 'LS'” but does not mention **UMOD**. Add “or 'UMOD'” for accuracy.

### 3.6 saveConfigToHeader (auxiliary/_auxiliary.py)

- “Save the used **section** of the MasterConfig” is ambiguous: it writes key-value pairs from a config **subsection** (e.g. config["KIN"]) into the FITS header. Values are written with `hdu.header[i] = config[i]`, which can fail for non-scalar or non-string values. Docstring could state “writes each key-value pair of the given config subsection into the HDU header” and that values must be header-compatible.

### 3.7 convertConfigDataType (initialise/_initialise.py)

- **Dead code:** The function is never called (only defined). YAML loading already produces Python types. Either remove it or integrate it and document where config types are normalized.
- If kept: docstring says “Convert … to the most suitable data type”; the code does not handle `value is None` (e.g. `value.lower()` would raise). Add a None check if the function is ever used.

---

## 4. Code quality and robustness

### 4.1 Bare `except` clauses

- Multiple files use `except:` or `except: pass`, which catches all exceptions (including `KeyboardInterrupt`, `SystemExit`) and can hide bugs. Files include:
  - `initialise/_initialise.py` (convertConfigDataType),
  - `stellarKinematics/ppxf_kin_wrapper.py`,
  - `continuumCube/ppxf_cont_wrapper.py`,
  - `emissionLines/ppxf_gas_wrapper.py`, `gandalf_gas_wrapper.py`,
  - `prepareSpectra/default.py`,
  - `lineStrengths/default.py`,
  - `userModules/twocomp_ppxf.py`,
  - and others (see grep for “except:”).
- **Recommendation:** Replace with `except Exception:` (and optionally log or re-raise) so that system exits and keyboard interrupts are not swallowed. In ppxf wrappers, document that failed fits return NaN and that exceptions are caught for robustness.

### 4.2 YAML loaders (initialise/_initialise.py)

- `readMasterConfig` uses `yaml.safe_load`; `loadConfig` uses `yaml.FullLoader`. Both are safe for untrusted input; for consistency and clarity, prefer `yaml.safe_load` in both, or document why FullLoader is required in one place.

### 4.3 Unused imports (MainPipeline.py)

- `ascii`, `fits`, `interp1d`, `time` are imported but not used. Remove them to avoid confusion and to satisfy static checkers.

---

## 5. Summary table

| Category        | Item                                      | Severity   | Location / note                          |
|-----------------|-------------------------------------------|------------|------------------------------------------|
| Bug             | CONT serial mode NameError                | Critical   | continuumCube/ppxf_cont_wrapper.py       |
| Bug             | addPathsToConfig IndexError               | High       | initialise/_initialise.py                |
| Bug             | getLSF UMOD should be elif                | Medium     | auxiliary/_auxiliary.py                  |
| Scientific      | EW extrapolation out of wavelength range  | Low        | emissionLines (ppxf/gandalf)             |
| Scientific      | BIN_ID contiguous assumption for KIN cont | Low        | emissionLines _load_kin_stellar_continuum|
| Scientific      | len(ubins) vs n_bins in saveContLineCube   | Low        | writeFITS/save_maps_fits.py               |
| Docstring       | “kinematics” → “CONT” for continuum cube  | Medium     | writeFITS/save_maps_fits.py              |
| Docstring       | readMasterConfig galindex unused          | Low        | initialise/_initialise.py                |
| Docstring       | getLSF docstring missing UMOD             | Low        | auxiliary/_auxiliary.py                   |
| Comment         | GIST vs nGIST in MainPipeline             | Low        | MainPipeline.py                          |
| Comment         | “Amrlia” typo                             | Trivial    | initialise/_initialise.py                |
| Dead code       | numberOfGalaxies, convertConfigDataType  | Low        | MainPipeline.py, initialise               |
| Code quality    | Bare except                               | Medium     | Multiple files                           |
| Code quality    | Unused imports                            | Low        | MainPipeline.py                          |

---

## 6. Recommendations (priority order)

1. **Fix CONT serial-mode bug** (move or remove `os.remove` in continuumCube/ppxf_cont_wrapper.py).
2. **Harden addPathsToConfig** (guard `len(line) >= 2` in initialise/_initialise.py).
3. **Correct getLSF** (use `elif` for UMOD in auxiliary/_auxiliary.py).
4. **Update misleading comments** in save_maps_fits.py (CONT, not kinematics).
5. **Replace bare `except`** with `except Exception` (and logging) in ppxf/continuum/emission wrappers and related modules.
6. **Tighten EW behaviour** when line is outside wavelength range (clip or NaN).
7. **Remove or use** dead code (numberOfGalaxies, convertConfigDataType) and clean unused imports in MainPipeline.py.
8. **Document** BIN_ID 0-based contiguous assumption where KIN continuum is used for EW.

---

*End of report. This review is intended to support scientific integrity and maintainability; address critical and high-severity items before production use.*
