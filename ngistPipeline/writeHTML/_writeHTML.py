import logging
import os

from printStatus import printStatus

import numpy as np
from ngistPipeline.mapviewer import loadData as _loadData
from ngistPipeline.writeHTML.generate_java import *
from ngistPipeline.writeHTML.plotlyData import *
from ngistPipeline.writeHTML.loadlyData import *


def find_nearest_idx(arrays, target_value):
    absolute_diff = np.abs(arrays - target_value)
    nearest_index = absolute_diff.argmin()
    return nearest_index

def writeHTML(config):
    """
    generateFITS _summary_

    Args:
        config (_type_): _description_
        module (_type_): _description_

    Returns:
        _type_: _description_
    """

    rootname = config["GENERAL"]["RUN_ID"]
    outdir = os.path.join(config["GENERAL"]["OUTPUT"], 'mapviewer_web')
    if os.path.exists(outdir) == False: os.mkdir(outdir)


    try:
        printStatus.running("Producing nGIST maps in HTML format")
        database = gistDataBase()
        database.loadData(config["GENERAL"]["OUTPUT"])

        percentiles = [1, 99]
        percentile_suffixs = ['PERCENT1', 'PERCENT99']

        module_names = ["TABLE", "MASK", "KIN", "GAS", "SFH", "LS"]
        module_table_names = ["table", "Mask", "kinResults", "gasResults", "sfhResults", "lsResults"]
        plot_modules = ["KIN", "GAS", "SFH", "LS"]

        for module in plot_modules:
            if getattr(database, module) == True:
                module_table = getattr(database, module_table_names[module_names.index(module)])
                module_table_pd = getattr(database, module_table_names[module_names.index(module)] + '_Vorbin_df')
                maptype_list = module_table.names
                maptype_list = [x for x in maptype_list if "ERR" not in x]
                for maptype in maptype_list:
                    for i, percentile in enumerate(percentiles):
                        logging.info("Working on generating " + "galinspec_%s_%s_%s.html" % (module, maptype, percentile_suffixs[i]))
                        if maptype == 'V':
                            nearest_index = find_nearest_idx(np.abs(module_table_pd[maptype]), np.nanpercentile(np.abs(module_table_pd[maptype]), percentile))
                        else:
                            nearest_index = find_nearest_idx(module_table_pd[maptype], np.nanpercentile(module_table_pd[maptype], percentile))
                        database.idxBinShort = module_table_pd['BIN_ID'][nearest_index]
                        fig_result = make_html_subplots(database, module, maptype, percentile_suffixs[i], module_table_pd[maptype][nearest_index])
                        fig_result.write_html(outdir + "/galinspec_%s_%s_%s.html" % (module, maptype, percentile_suffixs[i]))

        # Update json
        html_files = os.listdir(outdir) # List HTML files in the directory
        html_files = sorted(html_files, key=lambda x: os.path.getmtime(os.path.join(outdir, x))) # Sort the list by last modified time
        modules, module_magtypes, percentiles = extract_values(html_files) # Extract unique modules, magtypes, and percentiles
        with open(outdir+"/main.js", "w") as js_file: # Write the JavaScript code to the JavaScript file
            js_code = generate_js_code(modules, module_magtypes, percentiles)
            js_file.write(js_code)

        # Write the main HTML to a file
        with open(outdir+"/main.html", "w", encoding="utf-8") as f:
            f.write(create_main(rootname))

        printStatus.updateDone("Producing nGIST maps in HTML format")
        logging.info("Produced nGIST maps in HTML format")

    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to produce nGIST maps in HTML format."
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    return None
