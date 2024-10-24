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
				"NGC0000Example_BinSpectra_linear.hdf5",
				"NGC0000Example_BinSpectra.hdf5",
				"NGC0000Example_gas_BIN.fits",
				"NGC0000Example_gas-bestfit_BIN.fits",
				"NGC0000Example_gas-cleaned_BIN.fits",
				"NGC0000Example_kin-bestfit.fits",
				"NGC0000Example_kin-optimalTemplates.fits",
				"NGC0000Example_kin.fits",
				"NGC0000Example_ls_AdapRes.fits",
				"NGC0000Example_ls_OrigRes.fits",
				"NGC0000Example_ls-cleaned_linear.fits",
				"NGC0000Example_mask.fits",
				"NGC0000Example_sfh-bestfit.fits",
				"NGC0000Example_sfh-weights.fits",
				"NGC0000Example_sfh.fits",
				"NGC0000Example_table.fits",
				"NGC0000Example_KIN_maps.fits",
	       		"NGC0000Example_gas_BIN_maps.fits",
				"NGC0000Example_SFH_maps.fits",
				"NGC0000Example_LS_ORIGINAL_maps.fits",
				"NGC0000Example_LS_ADAPTED_maps.fits"]

for file in output_files:
	assert os.path.isfile(f"{output_dir}/{file}"), f"{file} not created, probably the module failed"
with open(f"{output_dir}/LOGFILE") as f:
	for line in f:
		print(line)

map_files = ["NGC0000Example_sfh.pdf",
		"NGC0000Example_gas-[NI]_5200.39_A_BIN.pdf",
		"NGC0000Example_gas-[NI]_5200.39_F_BIN.pdf",
		"NGC0000Example_gas-[NI]_5200.39_S_BIN.pdf",
		"NGC0000Example_gas-[NI]_5200.39_V_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_A_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_F_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_S_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_V_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_A_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_F_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_S_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_V_BIN.pdf",
		"NGC0000Example_kin.pdf",
		"NGC0000Example_ls_AdapRes.pdf",
		"NGC0000Example_ls_OrigRes.pdf",
		"NGC0000Example_ls-spp.pdf"]

#for file in map_files:
#	assert os.path.isfile(f"{output_dir}/maps/{file}"), f"{file} not created, probably the module failed"









