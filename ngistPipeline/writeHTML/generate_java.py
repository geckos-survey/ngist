import os

def extract_values(files):
    '''
    # Function to extract unique values of modules, magtypes, and percentiles
    '''
    modules = []
    module_magtypes = {}
    percentiles = []

    for filename in files:


        parts = os.path.splitext(filename)[0].split("_")

        if len(parts) >= 4 and parts[0].startswith("galinspec"):


            module, magtype = parts[1], parts[2]
            if len(parts) > 4:
                magtype += "_" + "_".join(parts[3:-1])
            percentile = parts[-1]

            if module not in modules:
                modules.append(module)
            if module not in module_magtypes:
                module_magtypes[module] = []
            if magtype not in module_magtypes[module]:
                module_magtypes[module].append(magtype)
            if percentile not in percentiles:
                percentiles.append(percentile)


    return list(modules), {module: list(magtypes) for module, magtypes in module_magtypes.items()}, list(percentiles)




# Function to generate JavaScript code
def generate_js_code(modules, module_magtypes, percentiles):
    '''
    Generate a JSON file representing the module tree structure of nGIST results.
    '''
    js_code = f"""
    document.addEventListener("DOMContentLoaded", function() {{
        const modules = {str(modules)};
        const moduleMagtypes = {module_magtypes};
        const percentileOptions = {str(percentiles)};

        const buttonsDiv = document.getElementById("buttons");
        const magtypeSelect = document.getElementById("magtypeSelect");
        const percentileSelect = document.getElementById("percentileSelect");

        let selectedModule = modules[0];
        let selectedMagtype = moduleMagtypes[selectedModule][0];
        let selectedPercentile = percentileOptions[0];

        const updateMagtypeOptions = (module) => {{
            const magtypeOptionsHTML = moduleMagtypes[module].map(magtype => `
                <option value="${{magtype}}">${{magtype}}</option>
            `).join("");
            magtypeSelect.innerHTML = magtypeOptionsHTML;
        }};

        const updatePercentileOptions = () => {{
            const percentileOptionsHTML = percentileOptions.map(percentile => `
                <option value="${{percentile}}">${{percentile}}</option>
            `).join("");
            percentileSelect.innerHTML = percentileOptionsHTML;
        }};

        window.changeModule = function(module) {{
            selectedModule = module;
            selectedMagtype = moduleMagtypes[module][0];
            selectedPercentile = percentileOptions[0];
            updateMagtypeOptions(module);
            updatePercentileOptions();
            updateContent(selectedModule, selectedMagtype, selectedPercentile);
        }};

        // Event listener for changes in the "magtype" dropdown
        magtypeSelect.addEventListener("change", function() {{
            selectedMagtype = magtypeSelect.value;
            updateContent(selectedModule, selectedMagtype, selectedPercentile);
        }});

        // Event listener for changes in the "percentile" dropdown
        percentileSelect.addEventListener("change", function() {{
            selectedPercentile = percentileSelect.value;
            updateContent(selectedModule, selectedMagtype, selectedPercentile);
        }});

        function updateContent(module, magtype, percentile) {{
            const src = `galinspec_${{module}}_${{magtype}}_${{percentile}}.html`;
            const iframe = document.getElementById("iframe");
            iframe.src = src;
        }}

        // Populate the buttons for selecting modules
        buttonsDiv.innerHTML = modules.map(module => `
            <button onclick="changeModule('${{module}}')">${{module}}</button>
        `).join("");

        updateMagtypeOptions(selectedModule);
        updatePercentileOptions();
        updateContent(selectedModule, selectedMagtype, selectedPercentile);
    }});
    """

    return js_code


def create_main(run_name):
    '''
    Generate the main HTML file
    '''
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>{run_name}</title>
    </head>
    <body>
        <h1>{run_name}</h1>
    
        <!-- Buttons for selecting modules -->
        <div id="buttons">
            <!-- Buttons for all available modules will be populated by JavaScript -->
        </div>
    
        <!-- Dropdowns for selecting "magtype" and "percentile" -->
        <select id="magtypeSelect">
            <!-- Dropdown options for "magtype" will be populated by JavaScript -->
        </select>
        
        <select id="percentileSelect">
            <!-- Dropdown options for "percentile" will be populated by JavaScript -->
        </select>
    
        <div id="content">
            <!-- Initial content -->
            <iframe id="iframe" src="galinspec_KIN_r_10_percentile1.html" width="100%" height="700"></iframe>
        </div>
    
        <script src="main.js"></script>
    </body>
    </html>
    """

    return html_content
