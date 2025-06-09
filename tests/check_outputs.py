import os

# home_dir = os.path.expanduser('~')

# print(home_dir)
output_dir = f"./.github/workflows/tests/gistTutorial/results/NGC0000Example"

with open(f"{output_dir}/LOGFILE") as f:
	for line in f:
		print(line)
print('I got up to here!')
#Check the output directory is created 
with open(f"{output_dir}/LOGFILE") as f:
	for line in f:
		print(line)
assert os.path.isdir(output_dir), "Output dir not created"
#assert os.path.isdir(f"{output_dir}/maps"), "Maps dir not created"



#Check the pipeline creates the correct files
output_files = ["CONFIG","LOGFILE",
				# "NGC0000Example_AllSpectra.fits", # no longer outputted
				"NGC0000Example_bin_spectra_linear.hdf5",
				"NGC0000Example_bin_spectra.hdf5",
				"NGC0000Example_gas_bin.fits",
				"NGC0000Example_gas_bestfit_bin.fits",
				"NGC0000Example_gas_cleaned_bin.fits",
				"NGC0000Example_kin_bestfit.fits",
				"NGC0000Example_kin_optimal_templates.fits",
				"NGC0000Example_kin.fits",
				"NGC0000Example_ls_adap_res.fits",
				"NGC0000Example_ls_orig_res.fits",
				"NGC0000Example_ls_cleaned_linear.fits",
				"NGC0000Example_mask.fits",
				"NGC0000Example_sfh_bestfit.fits",
				"NGC0000Example_sfh_weights.fits",
				"NGC0000Example_sfh.fits",
				"NGC0000Example_table.fits",
				"NGC0000Example_kin_maps.fits",
	       		"NGC0000Example_gas_bin_maps.fits",
				"NGC0000Example_sfh_maps.fits",
				"NGC0000Example_ls_original_maps.fits",
				"NGC0000Example_ls_adapted_maps.fits"]

for file in output_files:
	assert os.path.isfile(f"{output_dir}/{file}"), f"{file} not created, probably the module failed"
with open(f"{output_dir}/LOGFILE") as f:
	for line in f:
		print(line)

map_files = ["NGC0000Example_sfh.pdf",
		"NGC0000Example_gas_[NI]_5200.39_A_bin.pdf",
		"NGC0000Example_gas_[NI]_5200.39_F_bin.pdf",
		"NGC0000Example_gas_[NI]_5200.39_S_bin.pdf",
		"NGC0000Example_gas_[NI]_5200.39_V_bin.pdf",
		"NGC0000Example_gas_[OIII]_5006.77_A_bin.pdf",
		"NGC0000Example_gas_[OIII]_5006.77_F_bin.pdf",
		"NGC0000Example_gas_[OIII]_5006.77_S_bin.pdf",
		"NGC0000Example_gas_[OIII]_5006.77_V_bin.pdf",
		"NGC0000Example_gas_Hb_4861.32_A_bin.pdf",
		"NGC0000Example_gas_Hb_4861.32_F_bin.pdf",
		"NGC0000Example_gas_Hb_4861.32_S_bin.pdf",
		"NGC0000Example_gas_Hb_4861.32_V_bin.pdf",
		"NGC0000Example_kin.pdf",
		"NGC0000Example_ls_adap_res.pdf",
		"NGC0000Example_ls_orig_res.pdf",
		"NGC0000Example_ls_spp.pdf"]

#for file in map_files:
#	assert os.path.isfile(f"{output_dir}/maps/{file}"), f"{file} not created, probably the module failed"
