# nGIST Pipeline: Architecture and Usage

This document describes the **architecture** and **usage** of the nGIST pipeline. It is structured for both human readers and agentic AI: sections use consistent headings, bullet lists, and key-value style so that tools can locate entry points, config keys, modules, and data flow.

**Related documents**

- [PRODUCTS.md](PRODUCTS.md) — Data products (file naming, format, structure, units).
- [https://geckos-survey.github.io/gist-documentation/](https://geckos-survey.github.io/gist-documentation/) — Full documentation (installation, configuration, tutorials).

---

## Quick reference (for agentic AI)

| Topic | Value |
|-------|--------|
| **Main entry point** | `ngistPipeline` (console script) → `ngistPipeline.MainPipeline:main` |
| **Config** | Single YAML file path passed via `--config`; optional `--default-dir` for base paths |
| **Run model** | One run per invocation; one galaxy per config file (`ngalaxies = 1`) |
| **Config keys (top-level)** | `GENERAL`, `READ_DATA`, `SPATIAL_MASKING`, `SPATIAL_BINNING`, `PREPARE_SPECTRA`, `PREPARE_TEMPLATES`, `KIN`, `CONT`, `GAS`, `SFH`, `LS`, `UMOD` (optional) |
| **Output root** | `config["GENERAL"]["OUTPUT"]`; run prefix = `os.path.join(OUTPUT, config["GENERAL"]["RUN_ID"])` |
| **Main source** | `ngistPipeline/MainPipeline.py` (orchestration); modules under `ngistPipeline/<module_name>/` |
| **Module pattern** | Each pipeline stage has `_<module>/_<module>.py` with `<module>_Module(config[, cube])`; method chosen by `config["<SECTION>"]["METHOD"]`; plugins are `<Method>.py` or `<method>_<suffix>_wrapper.py` in the same folder |

**Pipeline stages (order)**

1. initialise  
2. readData  
3. spatialMasking  
4. spatialBinning  
5. prepareSpectra  
6. stellarKinematics (KIN)  
7. continuumCube (CONT)  
8. emissionLines (GAS)  
9. starFormationHistories (SFH)  
10. lineStrengths (LS)  
11. userModules (UMOD)

**Other entry points**

- `Mapviewer` → `ngistPipeline.Mapviewer:main` (GUI)
- `gistPlot_kin`, `gistPlot_gas`, `gistPlot_sfh`, `gistPlot_ls` → plotting scripts under `ngistPipeline.plotting`

---

## 1. High-level overview

The nGIST pipeline is a **sequential, single-galaxy** IFS analysis pipeline. Each run:

1. **Loads** one datacube and its metadata (readData).
2. **Prepares** the field: spatial mask, Voronoi binning, extraction of binned (and optionally spaxel) spectra into HDF5 (spatialMasking, spatialBinning, prepareSpectra).
3. **Runs analysis modules** in a fixed order: stellar kinematics (KIN), continuum cube (CONT), emission lines (GAS), star formation histories (SFH), line strengths (LS), and optional user modules (UMOD). Each module reads from the output directory and/or HDF5/FITS produced by earlier steps and writes FITS (and sometimes HDF5) products.
4. **Writes** all final maps and tables via the writeFITS layer, invoked from each analysis module.

There is **no in-pipeline parallelisation across galaxies**; one process runs one config file. Within a run, modules may use multiprocessing (e.g. per bin) as configured.

---

## 2. Execution flow

**Entry:** `main()` in `ngistPipeline/MainPipeline.py`.

- **CLI:** `ngistPipeline --config <path_to_config.yaml> [--default-dir <path_to_defaultDir>]`
- **Required:** `--config` must point to an existing YAML file.
- **Optional:** `--default-dir` points to a text file that defines base directories (see Configuration).

**Per run (single galaxy):**

1. **Initialise**  
   - `_initialise.readMasterConfig(configFile)` → config dict (YAML).  
   - `_initialise.addPathsToConfig(config, dirPath)` → merges paths from `defaultDir` file if provided.  
   - `_initialise.checkOutputDirectory(config)` → creates output dir if needed; saves or checks `CONFIG` in output.  
   - `_initialise.setupLogfile(config)` → log file in output directory.  
   - Any failure here exits before loading data.

2. **Preparation (data-dependent; cube in memory)**  
   - `_readData.readData_Module(config)` → returns `cube` dict or `"SKIP"`.  
   - `_spatialMasking.spatialMasking_Module(config, cube)` → writes mask FITS.  
   - `_spatialBinning.spatialBinning_Module(config, cube)` → writes bin table FITS and triggers writeFITS for binned maps.  
   - `_prepareSpectra.prepareSpectra_Module(config, cube)` → writes HDF5 spectra (binned, and optionally all spaxels).  
   - `cube` is then dropped from memory.

3. **Analysis (config and files only)**  
   Each of the following runs in order; if a module returns `"SKIP"`, the pipeline calls `skipGalaxy(config)` and exits without running later modules.  
   - `_stellarKinematics.stellarKinematics_Module(config)`  
   - `_continuumCube.continuumCube_Module(config)`  
   - `_emissionLines.emissionLines_Module(config)`  
   - `_starFormationHistories.starFormationHistories_Module(config)`  
   - `_lineStrengths.lineStrengths_Module(config)`  
   - `_userModules.user_Modules(config)`  

4. **Finalise**  
   - Log success and exit.

**Thread/process limits:** At the very top of `MainPipeline.py`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `OMP_NUM_THREADS` are set to `1` to avoid over-subscription when using multiprocessing inside modules.

---

## 3. Module architecture

### 3.1 Dispatcher pattern

Each pipeline stage is implemented as a **dispatcher** that:

1. Reads the method name from config, e.g. `config["READ_DATA"]["METHOD"]` or `config["KIN"]["METHOD"]`.
2. Dynamically imports a plugin module from the same package (e.g. `readData/MUSE_WFM.py` or `stellarKinematics/ppxf_kin_wrapper.py`).
3. Calls a single entry function (e.g. `readCube(config)` or `extractStellarKinematics(config)`).
4. For analysis modules, after the plugin returns, the dispatcher calls `_writeFITS.generateFITS(config, '<MODULE_TAG>')` to produce FITS maps/tables.

**SKIP behaviour:** If the plugin or writeFITS raises, the dispatcher logs, returns `"SKIP"`, and MainPipeline exits the run. If `METHOD` is `False`, the module is turned off (no plugin run) and returns `None`; pipeline continues.

### 3.2 Plugin locations and entry points

| Stage | Config section | METHOD value → plugin file | Entry function / notes |
|-------|----------------|----------------------------|-------------------------|
| readData | READ_DATA | e.g. `MUSE_WFM` → `readData/MUSE_WFM.py` | `readCube(config)` → returns `cube` dict |
| spatialMasking | SPATIAL_MASKING | e.g. `default` → `spatialMasking/default.py` | `generateSpatialMask(config, cube)` |
| spatialBinning | SPATIAL_BINNING | e.g. `voronoi` → `spatialBinning/voronoi.py` | `generateSpatialBins(config, cube)`; then writeFITS SPATIAL_BINNING |
| prepareSpectra | PREPARE_SPECTRA | e.g. `default` → `prepareSpectra/default.py` | `prepSpectra(config, cube)` |
| stellarKinematics | KIN | e.g. `ppxf` → `stellarKinematics/ppxf_kin_wrapper.py` | `extractStellarKinematics(config)`; then writeFITS KIN |
| continuumCube | CONT | e.g. `ppxf` → `continuumCube/ppxf_cont_wrapper.py` | `createContinuumCube(config)`; then writeFITS CONT |
| emissionLines | GAS | e.g. `ppxf`, `gandalf`, `magpi_gandalf` → `emissionLines/<method>_gas_wrapper.py` | Emission routine; then writeFITS GAS |
| starFormationHistories | SFH | e.g. `ppxf` → `starFormationHistories/ppxf_sfh_wrapper.py` | `extractStarFormationHistories(config)`; then writeFITS SFH |
| lineStrengths | LS | e.g. `default` → `lineStrengths/default.py` | `measureLineStrengths(config)`; then writeFITS LS |
| userModules | UMOD | e.g. `twocomp_ppxf` → `userModules/twocomp_ppxf.py` | User-defined; then writeFITS UMOD |

**Shared sub-components**

- **prepareTemplates** is not a pipeline stage; it is called **by** the analysis plugins (KIN, CONT, GAS, SFH, userModules). `prepareTemplates/_prepareTemplates.py` dispatches to a routine (e.g. `miles.py`) based on `config[module_used]["TEMPLATE_SET"]`; that routine uses `config[module_used]["LIBRARY"]` and `config["GENERAL"]["TEMPLATE_DIR"]`. So stellar (and gas) template sets are configured per analysis module or via a shared PREPARE_TEMPLATES block.
- **writeFITS** is used only from the dispatchers: `writeFITS/_writeFITS.py` → `generateFITS(config, module)`; actual writing is in `writeFITS/save_maps_fits.py` (e.g. `savefitsmaps`, `saveContLineCube`, `savefitsmaps_GASmodule`, etc.).

### 3.3 Output reuse and overwrite

- **OW_OUTPUT** (`config["GENERAL"]["OW_OUTPUT"]`): if `False`, each analysis module checks for existing output files (e.g. `{RUN_ID}_kin.fits`, `{RUN_ID}_kin-bestfit.fits`). If all expected files exist, the module is skipped (no plugin run, no writeFITS). If `True`, existing files are overwritten.
- **OW_CONFIG** (`config["GENERAL"]["OW_CONFIG"]`): if `False`, a previously saved `CONFIG` in the output directory is loaded and compared to the current config; if `True`, the current config is written over the saved one.

---

## 4. Configuration

### 4.1 Config file (YAML)

- **Format:** YAML; top-level keys are section names.
- **Typical sections:**  
  `GENERAL`, `READ_DATA`, `SPATIAL_MASKING`, `SPATIAL_BINNING`, `PREPARE_SPECTRA`, `PREPARE_TEMPLATES`, `KIN`, `CONT`, `GAS`, `SFH`, `LS`; optional `UMOD`.

**GENERAL (representative keys)**

| Key | Meaning |
|-----|---------|
| RUN_ID | Run identifier; prefix for all output filenames and subdirectory under OUTPUT. |
| INPUT | Input filename (or path) relative to `inputDir` from defaultDir. |
| OUTPUT | Output path relative to `outputDir` from defaultDir; actual run output is `outputDir/OUTPUT/RUN_ID/`. |
| REDSHIFT | Redshift used for rest-frame. |
| PARALLEL, NCPU | Multiprocessing. |
| LSF_DATA, LSF_TEMP | LSF filenames relative to configDir. |
| OW_CONFIG | If True, overwrite saved CONFIG in output. |
| OW_OUTPUT | If True, overwrite existing FITS/HDF5. |

**Section-specific METHOD and options**

- Each stage that uses a plugin has a `METHOD` (or equivalent) naming the plugin (e.g. `READ_DATA.METHOD`: `MUSE_WFM`; `KIN.METHOD`: `ppxf`). Other keys (e.g. `LMIN`, `LMAX`, `TARGET_SNR`, `EMI_FILE`) are documented in the full docs and in the example configs under `tests/gistTutorial/configFiles/` (e.g. `config.yaml`, `MasterConfig.yaml`).

### 4.2 defaultDir file (optional)

- **Usage:** Pass path via `--default-dir`. If the file exists, `addPathsToConfig` reads it and overrides base paths in `config["GENERAL"]`.
- **Format:** Plain text; one assignment per line; comments with `#`.  
  - `inputDir=<path>`  
  - `outputDir=<path>`  
  - `configDir=<path>`  
  - `templateDir=<path>`  
  - Special: `configDir=outputDir` means use the output directory as config directory.
- Paths in the YAML (INPUT, OUTPUT, LSF_*, LIBRARY, etc.) are then interpreted relative to these bases (e.g. INPUT relative to `inputDir`, LSF relative to `configDir`, template LIBRARY relative to `templateDir`).

---

## 5. Data flow (conceptual)

1. **readData**  
   - Input: config, path from GENERAL.INPUT + inputDir.  
   - Output: in-memory `cube` (flux, variance, wavelength, header, mask, etc.). Some readData modules also set `cube["bunit"]` from the FITS BUNIT keyword.

2. **spatialMasking**  
   - Input: config, cube.  
   - Output: `{RUN_ID}_mask.fits` (which spaxels are masked).

3. **spatialBinning**  
   - Input: config, cube.  
   - Output: `{RUN_ID}_table.fits` (spaxel–bin mapping, SNR, etc.) and binned maps FITS written via writeFITS.

4. **prepareSpectra**  
   - Input: config, cube.  
   - Output: `{RUN_ID}_BinSpectra.hdf5`, `{RUN_ID}_BinSpectra_linear.hdf5`; if GAS runs at SPAXEL level, also `{RUN_ID}_AllSpectra.hdf5`. Optional BUNIT on HDF5 from input.

5. **KIN**  
   - Input: config; reads binned spectra and bin table from output directory.  
   - Output: kinematics FITS, best-fit FITS, optimal templates FITS, etc., via writeFITS.

6. **CONT**  
   - Input: config; uses KIN results and binned spectra.  
   - Output: continuum cube, line cube, original cube FITS; optional BUNIT/CUNIT3 from input header.

7. **GAS**  
   - Input: config; continuum and spectra (bin or spaxel level).  
   - Output: emission-line tables and maps FITS via writeFITS.

8. **SFH**  
   - Input: config; reads spectra and templates.  
   - Output: SFH table and weights FITS via writeFITS.

9. **LS**  
   - Input: config; reads spectra and line-strength band config.  
   - Output: line-strength FITS (original and adapted resolution) via writeFITS.

10. **UMOD**  
    - Input: config; depends on user plugin.  
    - Output: user FITS via writeFITS.

Exact file names and structures are in [PRODUCTS.md](PRODUCTS.md).

---

## 6. Usage summary

**Run one galaxy**

```bash
ngistPipeline --config /path/to/config.yaml
```

**Run with base directories from a file**

```bash
ngistPipeline --config /path/to/config.yaml --default-dir /path/to/defaultDir
```

**Typical workflow**

1. Install the package (see documentation).
2. Prepare input cube, mask (if any), LSF files, and template library paths.
3. Copy or adapt a config YAML (e.g. from `tests/gistTutorial/configFiles/`) and set RUN_ID, INPUT, OUTPUT, REDSHIFT, and module options.
4. Optionally create a defaultDir file and pass it with `--default-dir`.
5. Run `ngistPipeline --config <config>.yaml [--default-dir <defaultDir>]`.
6. Check output directory for CONFIG, LOGFILE, and all `{RUN_ID}_*` products (see PRODUCTS.md).
7. Use Mapviewer or gistPlot_* scripts to inspect results.

**Mapviewer and plotting**

- **Mapviewer:** `Mapviewer` (GUI) — loads an output directory and visualises pipeline products.
- **Plotting scripts:** `gistPlot_kin`, `gistPlot_gas`, `gistPlot_sfh`, `gistPlot_ls` — generate plots from pipeline outputs; see package docs for arguments.

---

## 7. File and directory layout (source)

| Path | Purpose |
|------|---------|
| `ngistPipeline/MainPipeline.py` | CLI and run orchestration. |
| `ngistPipeline/initialise/` | Config read, path merge, log and CONFIG write. |
| `ngistPipeline/readData/` | Dispatcher + plugins (MUSE_*, CALIFA_*, PLAINTXT, etc.). |
| `ngistPipeline/spatialMasking/` | Dispatcher + default (and optional) plugins. |
| `ngistPipeline/spatialBinning/` | Dispatcher + voronoi (and optional) plugins. |
| `ngistPipeline/prepareSpectra/` | Dispatcher + default plugin. |
| `ngistPipeline/prepareTemplates/` | Called by analysis plugins; miles, walcher, bpass, etc. |
| `ngistPipeline/stellarKinematics/` | Dispatcher + ppxf_kin_wrapper. |
| `ngistPipeline/continuumCube/` | Dispatcher + ppxf_cont_wrapper. |
| `ngistPipeline/emissionLines/` | Dispatcher + ppxf_gas_wrapper, gandalf_gas_wrapper, MAGPI_gandalf_gas_wrapper. |
| `ngistPipeline/starFormationHistories/` | Dispatcher + ppxf_sfh_wrapper. |
| `ngistPipeline/lineStrengths/` | Dispatcher + default, lsindex_spec, ssppop_fitting. |
| `ngistPipeline/userModules/` | Dispatcher + user plugins (e.g. twocomp_ppxf). |
| `ngistPipeline/writeFITS/` | generateFITS dispatcher + save_maps_fits (all FITS writing). |
| `ngistPipeline/auxiliary/` | Shared helpers (e.g. LSF loading, robust_sigma). |
| `ngistPipeline/plotting/` | gistPlot_* scripts. |
| `ngistPipeline/mapviewer/` | Mapviewer GUI. |
| `ngistPipeline/utils/` | e.g. WCS helpers. |

This layout and the dispatcher pattern allow adding or swapping plugins (e.g. a new readData method or a new GAS method) by adding a new file and setting the corresponding METHOD in the config.
