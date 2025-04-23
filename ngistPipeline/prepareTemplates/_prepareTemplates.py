import importlib.util
import logging
import os

from printStatus import printStatus


def prepareTemplates_Module(
    config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid=False
):
    """
    This function calls the prepareTemplates routine specified by the user.
    """
    # Import the chosen prepareTemplates routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config[module_used]["TEMPLATE_SET"]
            + ".py",
        )
        logging.info(
            "Using the routine for '" + config[module_used]["TEMPLATE_SET"] + ".py'"
        )
        prepTemplatesModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepTemplatesModule)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the routine '"
            + config[module_used]["TEMPLATE_SET"]
            + ".py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen prepareTemplates routine
    try:
        (
            templates,
            lamRange_spmod,
            logLam2,
            ntemplates,
            logAge_grid,
            metal_grid,
            alpha_grid,
            ncomb,
            nAges,
            nMetal,
            nAlpha,
        ) = prepTemplatesModule.prepareSpectralTemplateLibrary(
            config,
            lmin,
            lmax,
            velscale,
            LSF_Data,
            LSF_Templates,
            module_used,
            sortInGrid,
        )
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "Routine '" + config[module_used]["TEMPLATE_SET"] + ".py' failed."
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Return
    return (
        templates,
        lamRange_spmod,
        logLam2,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    )
