# Reassessment of v7.4.0 EW/KIN and Test Changes

This document reassesses all changes introduced for equivalent width (EW) using the stellar kinematics (KIN) module, and the associated tests, to ensure they are valid and will stand up to scrutiny.

---

## 1. Equivalent width (EW) calculation

### 1.1 Definition and convention

- **Definition**: EW = −F_line / f_cont(λ_line), where F_line is integrated line flux and f_cont is continuum flux density at the line centre.
- **Convention**: Negative EW for emission (flux above continuum), positive for absorption. This matches standard usage (e.g. Kennicutt 1992; Westfall et al. 2019 MaNGA DAP; Cappellari 2017 pPXF).

**Verdict**: Correct and consistent with literature.

### 1.2 Use of KIN stellar continuum

- When `_kin-bestfit.fits` exists and bin counts match, EW uses the **stellar kinematics (KIN) best-fit continuum** for f_cont.
- Otherwise, f_cont is taken from the gas-fit continuum (bestfit − gas_bestfit for pPXF; bestfit − emissionSpectra for GandALF).

**Rationale**: EW is a property of the spectrum relative to the stellar continuum. Using the KIN module’s stellar-only fit (no emission lines) gives a single, consistent definition of “stellar continuum” across the pipeline and avoids double-use of the gas fit.

**Verdict**: Scientifically sound; fallback preserves behaviour when KIN is missing or incompatible.

### 1.3 Interpolation and binning

- KIN bestfit is on the KIN wavelength grid; gas uses a (possibly different) LMIN/LMAX. The code **interpolates** the KIN continuum onto the gas wavelength grid (linear in wavelength).
- For **BIN** level: one continuum spectrum per bin; shape (nbins, npix_gas).
- For **SPAXEL** level: continuum is expanded from bins to spaxels via `BIN_ID` in `_table.fits`; shape (n_spaxels, npix_gas).
- Missing file, shape mismatch, or bin-count mismatch result in fallback to gas-fit continuum (no hard failure).

**Verdict**: Implementation is consistent with data layout and safe against missing/mismatched KIN.

### 1.4 Edge cases

- **Negative BIN_ID**: Only spaxels with `bin_id >= 0` get KIN continuum; others get NaN and then NaN EW (handled by existing EW logic).
- **Zero/negative f_cont**: EW set to NaN; no division by zero.
- **Continuum error**: EW uncertainty uses local scatter in continuum; acceptable for current use.

**Verdict**: Edge cases are handled; no unguarded divisions or invalid arrays.

---

## 2. Code changes

### 2.1 `_load_kin_stellar_continuum()`

- **Location**: `ppxf_gas_wrapper.py` and `gandalf_gas_wrapper.py`.
- **Behaviour**: Loads `_kin-bestfit.fits` (BESTFIT + LOGLAM), interpolates to gas wavelengths, optionally expands to spaxels. Returns `None` on missing file, read error, or bin-count mismatch.
- **Dependencies**: Only `os`, `numpy`, `astropy.io.fits`; no extra I/O or side effects.

**Verdict**: Clear, self-contained, and fail-safe.

### 2.2 `compute_equivalent_width()` / `compute_equivalent_width_gandalf()`

- **Change**: Optional argument `stellar_continuum=None`. When provided and shape matches (n_spectra, npix), it is used for f_cont; otherwise the previous behaviour (bestfit − gas/emission) is used.
- **Backward compatibility**: Callers that do not pass `stellar_continuum` behave exactly as before.

**Verdict**: Backward compatible and minimal.

### 2.3 Call sites

- Both wrappers call `_load_kin_stellar_continuum()` before computing EW and pass the result into the EW routine. No change to other pipeline steps.

**Verdict**: Integration is localised and correct.

---

## 3. Outputs and FITS columns

- **pPXF gas table**: For each line, columns `{NAME}_EW`, `{NAME}_EW_ERR`, `{NAME}_CONT` (Angstrom, Angstrom, flux density).
- **GandALF**: Same semantics; column naming follows existing GandALF pattern (`{NAME}_{LAMBDA}_EW`, etc.).
- **EW maps**: Produced by existing map-writing logic that already reads the gas tables.

**Verdict**: Output schema is consistent and documented in CHANGELOG.

---

## 4. Tests

### 4.1 Unit tests (`tests/test_emission_ew.py`)

- **Without KIN**: EW from (bestfit − gas_bestfit); shapes and sign (negative for emission) checked.
- **With KIN**: Provided stellar continuum used; shapes and finite values checked.
- **KIN loader – missing file**: Returns `None`.
- **KIN loader – BIN**: With a temporary `_kin-bestfit.fits`, returns (nbins, npix_gas) and interpolates correctly.

**Verdict**: Core behaviour and loader are covered; tests are deterministic and do not depend on main pipeline data.

### 4.2 Integration check (`.github/workflows/tests/check_outputs.py`)

- Asserts gas BIN FITS has at least one `*_EW` and one `*_EW_ERR` column.
- Asserts at least one finite EW value.
- Asserts `_kin-bestfit.fits` exists and bin count matches gas table (so EW can use KIN when expected).

**Verdict**: Ensures EW and KIN are produced and used in the CI run.

### 4.3 CI workflow

- Unit tests run before the pipeline; integration check runs after. All steps use the same config (NGC 0000 example).

**Verdict**: Order and scope are appropriate.

---

## 5. Summary

| Area              | Status   | Notes                                                |
|-------------------|----------|------------------------------------------------------|
| EW definition     | Valid    | Standard convention; literature references in docstrings |
| KIN continuum use | Valid    | Clear rationale; fallback when KIN unavailable       |
| Interpolation     | Valid    | Linear in wavelength; correct BIN/SPAXEL handling    |
| Edge cases        | Valid    | NaN/zero handled; no unsafe operations              |
| Backward compat   | Valid    | Optional argument; unchanged behaviour if not used   |
| Outputs           | Valid    | Documented; consistent with existing FITS usage    |
| Tests             | Valid    | Unit + integration; CI runs full pipeline           |

**Overall**: The changes are scientifically and technically sound, backward compatible, and well covered by tests. They are suitable for production use after successful batch testing (e.g. on the MAUVE sample in the toby_sandbox).
