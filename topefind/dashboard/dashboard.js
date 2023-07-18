importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.4/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.4/dist/wheels/panel-0.14.4-py3-none-any.whl', 'pyodide-http==0.1.0', 'matplotlib', 'numpy', 'pandas', 'param', 'plotly', 'scipy']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

"""
Dashboard to visualize results and see comparisons between models.
This script currently does not envision any normal programming pattern, uses a large number of globals, and it will need
some optimization if used in a highly business critical perspective. Please use at your own risk and revise it priorly.


This script modality applies to both serving the components through panel itself and to
the ability to convert it with panel into a pyodide, pyodide-worker, or pyscript app.
It is easily convertible with: \`\`\`panel convert dashboard.py --to pyodide-worker --out . \`\`\` from the dashboard dir.
Set the global PYSCRIPT to True before the conversion.

If you want to use pyodide-worker remember to change the generated .js with the correct URLs to the data
you want to provide which has to be served on some service.

You can easily serve everything locally as well when running from pyodide-worker by starting a http server:
\`\`\` python3 -m http.server \`\`\` and then opening the dashboard.html in a browser.

Please check: https://panel.holoviz.org/how_to/wasm/convert.html for more details.
"""
import re
import copy
from pathlib import Path

import param
import panel as pn
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import rankdata
from panel.reactive import ReactiveHTML

# Globals
PYSCRIPT = True

STRUCTURE_EXT = "bcif"
ASSETS_NAME = "http://localhost:8000" if PYSCRIPT else "assets"
MODEL_COMP_PATH = f"{ASSETS_NAME}/benchmark.pkl.gz"

# To load from local, e.g. after pyscript fetch
# PDBE_JS_PATH = f"{ASSETS_NAME}/pdbe-molstar-plugin-3.1.1.js"
# PDBE_CSS_PATH = f"{ASSETS_NAME}/pdbe-molstar-light-3.1.1.css"
# PDBE_CSS_DARK_PATH = f"{ASSETS_NAME}/pdbe-molstar-3.1.1.css"

# To load from ebi:
PDBE_JS_PATH = "https://www.ebi.ac.uk/pdbe/pdb-component-library/js/pdbe-molstar-plugin-3.1.1.js"
PDBE_CSS_PATH = "https://www.ebi.ac.uk/pdbe/pdb-component-library/css/pdbe-molstar-light-3.1.1.css"
PDBE_CSS_DARK_PATH = "https://www.ebi.ac.uk/pdbe/pdb-component-library/css/pdbe-molstar-3.1.1.css"

DF = pd.read_pickle(MODEL_COMP_PATH)
NON_SELECTED_COLOR = {"r": 155, "g": 155, "b": 155}
AG_COLOR = {"r": 10, "g": 200, "b": 0}

NORMAL_ROW_MAX_HEIGHT = 500
MODELS = DF["model"].unique().tolist()
PDBS = DF["pdb"].unique().tolist()
METRICS = DF["metric"].unique().tolist()
REGIONS = DF["region"].unique().tolist()
CHAINS = ["heavy", "light", "both"]
DIFF_MODES = ["abs_norm_diff", "abs_norm_rank_diff"]
PDBE_EXTRA_PARAMS = ["visual_style", "spin"]

PDBE_MOLSTAR = ReactiveHTML()
SCATTER_PLOT_PANEL = pn.pane.Plotly()
VIOLIN_PLOT_PANEL = pn.pane.Plotly()
MODELS_X_WIDGET = pn.widgets.Select(name='Model on x axis', options=MODELS)
MODELS_Y_WIDGET = pn.widgets.Select(name='Model on y axis', options=MODELS)
CHAINS_WIDGET = pn.widgets.Select(name='Chain', options=CHAINS)
METRICS_WIDGET = pn.widgets.Select(name='Metric', options=METRICS)
REGIONS_WIDGET = pn.widgets.Select(name='Region', options=REGIONS)
PDBS_WIDGET = pn.widgets.Select(name='PDB', options=PDBS)
DIFFERENCE_MODE_WIDGET = pn.widgets.Select(name='Difference Mode', options=DIFF_MODES)
PREV_PDB = PDBS[0]

PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "title_x": 0.5,
}
PLOTLY_AXES_STYLE = {
    "showgrid": True,
    "gridwidth": 0.6,
    "gridcolor": "lightgray",
    "zeroline": True,
    "zerolinewidth": 1,
    "zerolinecolor": "gray",
}

REPRESENTATIONS = [
    "cartoon",
    "ball-and-stick",
    "carbohydrate",
    "distance-restraint",
    "ellipsoid",
    "gaussian-surface",
    "molecular-surface",
    "point",
    "putty",
    "spacefill",
]


# Classes
class ProteinParatopeSingleController(param.Parameterized):
    show_paratope_labels = param.Action()
    show_paratope_predictions = param.Action()
    reset_colors = param.Action()
    full_reset = param.Action()

    def __init__(self, pdbe, df, model, chain, metric, pdb_id, region, **params):
        self.pdbe = pdbe
        self.model = model
        self.chain = chain
        self.metric = metric
        self.pdb_id = pdb_id
        self.region = region
        self.non_selected_color = NON_SELECTED_COLOR
        self.parsed_chain = ["heavy", "light"] if self.chain == "both" else [self.chain]
        self.df = df[
            (df["model"].isin([self.model])) &
            (df["pdb"].isin([self.pdb_id])) &
            (df["metric"].isin([self.metric])) &
            (df["region"].isin([self.region]))
            ]
        super().__init__(**params)

        self.show_paratope_labels = self._action_show_paratope_labels
        self.show_paratope_predictions = self._action_show_paratope_predictions
        self.reset_colors = self._action_reset_color
        self.full_reset = self._action_full_reset

    def _action_show_paratope_labels(self, _):
        para_labels_selections = find_paratope_labels(self.parsed_chain, self.df)
        self.pdbe.color(para_labels_selections, self.non_selected_color)

    def _action_show_paratope_predictions(self, _):
        para_preds_selections = find_paratope_preds(self.parsed_chain, self.df)
        self.pdbe.color(para_preds_selections, self.non_selected_color)

    def _action_reset_color(self, _):
        self.pdbe.clear_selection()

    def _action_full_reset(self, _):
        self.pdbe.clear_selection()
        reset_data = {
            "camera": True,
            "theme": True,
            "highlightcolor": True,
            "selectColor": True,
        }
        self.pdbe.reset(reset_data)


class ProteinParatopeOverlappedController(param.Parameterized):
    show_paratope_labels = param.Action()
    show_models_diff = param.Action()
    show_model_x_ground_truth_diff = param.Action()
    show_model_y_ground_truth_diff = param.Action()
    reset_colors = param.Action()
    full_reset = param.Action()

    def __init__(self, pdbe, df, model1, model2, chain, metric, pdb_id, region, diff_mode, **params):
        self.pdbe = pdbe
        self.model1 = model1
        self.model2 = model2
        self.chain = chain
        self.metric = metric
        self.pdb_id = pdb_id
        self.region = region
        self.diff_mode = diff_mode
        self.non_selected_color = {"r": 115, "g": 115, "b": 115}
        self.parsed_chain = ["heavy", "light"] if self.chain == "both" else [self.chain]

        self.df_model1 = df[
            (df["model"] == model1) &
            (df["pdb"] == self.pdb_id) &
            (df["metric"] == self.metric) &
            (df["region"] == self.region)
            ]
        self.df_model2 = df[
            (df["model"] == model2) &
            (df["pdb"] == self.pdb_id) &
            (df["metric"] == self.metric) &
            (df["region"] == self.region)
            ]

        super().__init__(**params)

        self.show_paratope_labels = self._action_show_paratope_labels
        self.show_models_diff = self._action_show_models_diff
        self.show_model_x_ground_truth_diff = self._action_show_model_x_ground_truth_diff
        self.show_model_y_ground_truth_diff = self._action_show_model_y_ground_truth_diff
        self.reset_colors = self._action_reset_color
        self.full_reset = self._action_full_reset

    def _action_show_paratope_labels(self, _):
        para_labels_selections = find_paratope_labels(self.parsed_chain, self.df_model1)
        self.pdbe.color(para_labels_selections, self.non_selected_color)

    def _action_show_models_diff(self, _):
        para_preds_diff = find_models_diffs(self.parsed_chain, self.df_model1, self.df_model2, self.diff_mode)
        self.pdbe.color(para_preds_diff, self.non_selected_color)

    def _action_show_model_x_ground_truth_diff(self, _):
        # Trick to use the same function find_models_diffs
        df2 = self.df_model2.drop(columns="predictions")
        df_cols = df2.columns
        df2.columns = [col if col != "full_paratope_labels" else "predictions" for col in df_cols]
        para_preds_diff = find_models_diffs(self.parsed_chain, self.df_model1, df2, self.diff_mode)
        self.pdbe.color(para_preds_diff, self.non_selected_color)

    def _action_show_model_y_ground_truth_diff(self, _):
        # Trick to use the same function find_models_diffs
        df1 = self.df_model1.drop(columns="predictions")
        df_cols = df1.columns
        df1.columns = [col if col != "full_paratope_labels" else "predictions" for col in df_cols]
        para_preds_diff = find_models_diffs(self.parsed_chain, df1, self.df_model2, self.diff_mode)
        self.pdbe.color(para_preds_diff, self.non_selected_color)

    def _action_reset_color(self, _):
        self.pdbe.clear_selection()

    def _action_full_reset(self, _):
        self.pdbe.clear_selection()
        reset_data = {
            "camera": True,
            "theme": True,
            "highlightcolor": True,
            "selectColor": True,
        }
        self.pdbe.reset(reset_data)


class PDBeMolStar(ReactiveHTML):
    """

    This is minified version of the panel-chemistry original PDBeMolStar with some minor modifications.
    The code in this file is vendored. Check the relative LICENSE and the original implementation:
    https://github.com/awesome-panel/panel-chemistry/blob/main/src/panel_chemistry/pane/pdbe_molstar.py

    Here is the original License:

    MIT License

        Copyright (c) 2021 Marc Skov Madsen

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    A Panel Pane to wrap the PDBe implementation of the Mol* ('MolStar') viewer.
    Check out
    - [PDBe Mol*](https://github.com/PDBeurope/pdbe-molstar)
    - [Mol*](https://molstar.org/)
    - [Mol* GitHub](https://github.com/molstar/molstar)
    Cite Mol*:
    David Sehnal, Sebastian Bittrich, Mandar Deshpande, Radka Svobodová, Karel Berka,
    Václav Bazgier, Sameer Velankar, Stephen K Burley, Jaroslav Koča, Alexander S Rose:
    Mol* Viewer: modern web app for 3D visualization and analysis of large biomolecular structures,
    Nucleic Acids Research, 2021; https://doi.org/10.1093/nar/gkab314.

    PDBe MolStar structure viewer.
    Set one of \`molecule_id\`, \`custom_data\` and \`ligand_view\`.
    For more information:
    - https://github.com/PDBeurope/pdbe-molstar/wiki
    - https://molstar.org/
    The implementation is based on the JS Plugin. See
    - https://github.com/PDBeurope/pdbe-molstar/wiki/1.-PDBe-Molstar-as-JS-plugin
    For documentation on the helper methods:
    - https://github.com/molstar/pdbe-molstar/wiki/3.-Helper-Methods
    See also https://embed.plnkr.co/plunk/m3GxFYx9cBjIanBp for an example JS
    """

    # Addition to the vendored version to allow for local deployment of the plugins.

    molecule_id = param.String(default=None, doc="PDB id to load. Example: '1qyn' or '1cbs'")

    custom_data = param.Dict(
        doc="""Load data from a specific data source. Example:
        { 
            "url": "https://www.ebi.ac.uk/pdbe/coordinates/1cbs/chains?entityId=1&asymId=A&encoding=bcif", 
            "format": "cif", 
            "binary": True 
        }
        """
    )
    ligand_view = param.Dict(
        doc="""This option can be used to display the PDBe ligand page 3D view
        like https://www.ebi.ac.uk/pdbe/entry/pdb/1cbs/bound/REA.
        Example: {"label_comp_id": "REA"}
        """
    )
    alphafold_view = param.Boolean(default=False, doc="Applies AF confidence score colouring theme for alphafold model")
    bg_color = param.Color("#F7F7F7", doc="""Color of the background. If \`None\`, color theme is applied""")
    assembly_id = param.String(doc="Specify assembly")
    highlight_color = param.Color(default="#ff6699", doc="Color for mouseover highlighting")
    select_color = param.Color(default="#0c0d11", doc="Color for selections")
    visual_style = param.Selector(default=None, objects=[None, *REPRESENTATIONS], doc="Visual styling")
    theme = param.Selector(default="default", objects=["default", "dark"], doc="CSS theme to use")
    hide_polymer = param.Boolean(default=False, doc="Hide polymer")
    hide_water = param.Boolean(default=False, doc="Hide water")
    hide_heteroatoms = param.Boolean(default=False, doc="Hide het")
    hide_carbs = param.Boolean(default=False, doc="Hide carbs")
    hide_non_standard = param.Boolean(default=False, doc="Hide non standard")
    hide_coarse = param.Boolean(default=False, doc="Hide coarse")
    hide_controls_icon = param.Boolean(default=False, doc="Hide the control menu")
    hide_expand_icon = param.Boolean(default=False, doc="Hide the expand icon")
    hide_settings_icon = param.Boolean(default=False, doc="Hide the settings menu")
    hide_selection_icon = param.Boolean(default=False, doc="Hide the selection icon")
    hide_animation_icon = param.Boolean(default=False, doc="Hide the animation icon")
    pdbe_url = param.String(default=None, constant=True, doc="Url for PDB data. Mostly used for internal testing")
    load_maps = param.Boolean(default=False, doc="Load electron density maps from the pdb volume server")
    validation_annotation = param.Boolean(default=False, doc="Adds 'annotation' control in the menu")
    domain_annotation = param.Boolean(default=False, doc="Adds 'annotation' control in the menu")
    low_precision_coords = param.Boolean(default=False, doc="Load low precision coordinates from the model server")
    hide_controls = param.Boolean(default=False, doc="Hide the control menu")
    sequence_panel = param.Boolean(default=True, doc="Show the sequence panel")
    expanded = param.Boolean(default=True, doc="""Display full-screen by default on load""")
    landscape = param.Boolean(default=True, doc="""Set landscape view.""")
    select_interaction = param.Boolean(default=True, doc="Switch on or off the default selection interaction behaviour")
    lighting = param.Selector(
        default="matte",
        objects=["flat", "matte", "glossy", "metallic", "plastic"],
        doc="Set the lighting",
    )
    default_preset = param.Selector(
        default="default",
        objects=["default", "unitcell", "all-models", "supercell"],
        doc="Set the preset view",
    )
    pdbe_link = param.Boolean(default=True, doc="Show the PDBe entry link at in the top right corner")
    spin = param.Boolean(default=False, doc="Toggle spin")
    _clear_highlight = param.Boolean(doc="Event to trigger clearing of highlights")
    _select = param.Dict(doc="Dictionary used for selections and coloring these selections")
    _clear_selection = param.Boolean(doc="Clear selection event trigger")
    _highlight = param.Dict(doc="Dictionary used for selections and coloring these selections")
    _reset = param.Boolean(doc="Reset event trigger")
    _args = param.Dict(doc="Dictionary with function call arguments")
    test = param.Boolean(default=False)

    # Test if template supports script loading:
    # <script type="text/javascript" src={PDBE_JS_PATH}> </script>
    _template = f"""
                <link id="molstarTheme" rel="stylesheet" type="text/css" href="{PDBE_CSS_PATH}"/>
                <div id="container" style="width:100%; height: 100%;"><div id="pdbeViewer"></div></div>
                """
    __javascript__ = [PDBE_JS_PATH]

    _scripts = {
        "render": r"""
        function standardize_color(str){
            var ctx = document.createElement("canvas").getContext("2d");
            ctx.fillStyle = str;
            return ctx.fillStyle;
        }
        function toRgb(color) {
          var hex = standardize_color(color)
          var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
          return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
          } : null;
        }
        state.toRgb = toRgb

        function getHideStructure(){
            var hideStructure = [];
            if (data.hide_polymer){hideStructure.push("polymer")}
            if (data.hide_water){hideStructure.push("water")}
            if (data.hide_heteroatoms){hideStructure.push("het")}
            if (data.hide_carbs){hideStructure.push("carbs")}
            if (data.hide_non_standard){hideStructure.push("nonStandard")}
            if (data.hide_coarse){hideStructure.push("coarse")}
            return hideStructure
        }

        function getHideCanvasControls(){
            var hideCanvasControls = [];
            if (data.hide_controls_icon){hideCanvasControls.push("controlToggle")}
            if (data.hide_expand_icon){hideCanvasControls.push("expand")}
            if (data.hide_settings_icon){hideCanvasControls.push("controlInfo")}
            if (data.hide_selection_icon){hideCanvasControls.push('selection')}
            if (data.hide_animation_icon){hideCanvasControls.push("animation")}
            return hideCanvasControls
        }

        state.getHideCanvasControls = getHideCanvasControls

        function getOptions(){
            var options = {
                moleculeId: data.molecule_id,
                customData: data.custom_data,
                ligandView: data.ligand_view,
                alphafoldView: data.alphafold_view,
                assemblyId: data.assembly_id,
                bgColor: toRgb(data.bg_color),
                highlightColor: toRgb(data.highlight_color),
                selectColor: toRgb(data.select_color),
                hideStructure: getHideStructure(),
                hideCanvasControls: getHideCanvasControls(),
                loadMaps: data.load_maps,
                validationAnnotation: data.validation_annotation,
                domainAnnotation: data.domain_annotation,
                lowPrecisionCoords: data.low_precision_coords,
                expanded: data.expanded,
                hideControls: data.hide_controls,
                landscape: data.landscape,
                selectInteraction: data.select_interaction,
                lighting: data.lighting,
                defaultPreset: data.default_preset,
                pdbeLink: data.pdbe_link,
                sequencePanel: data.sequence_panel,
            }
            if (data.visual_style!==null){
                options["visualStyle"]=data.visual_style
            }
            if (data.pdbe_url!==null){
                options["pdbeUrl"]=data.pdbe_url
            }
            return options
        }
        state.getOptions=getOptions
        self.theme()

        state.viewerInstance = new PDBeMolstarPlugin();
        state.viewerInstance.render(pdbeViewer, state.getOptions());

        """,
        "rerender": """
        state.viewerInstance.visual.update(state.getOptions(), fullLoad=true)
        """,
        "molecule_id": """self.rerender()""",
        "custom_data": """self.rerender()""",
        "ligand_view": """self.rerender()""",
        "alphafold_view": """self.rerender()""",
        "assembly_id": """self.rerender()""",
        "visual_style": """self.rerender()""",
        "bg_color": "state.viewerInstance.canvas.setBgColor(state.toRgb(data.bg_color))",
        "highlight_color": """
        state.viewerInstance.visual.setColor({highlight: state.toRgb(data.highlight_color)})""",
        "select_color": """
        state.viewerInstance.visual.setColor({select: state.toRgb(data.select_color)})""",
        "theme": f"""
        if (data.theme==="dark"){{
            molstarTheme.href="{PDBE_CSS_DARK_PATH}"
        }} else {{
            molstarTheme.href="{PDBE_CSS_PATH}"
        }}
        """,
        "hide_polymer": "state.viewerInstance.visual.visibility({polymer:!data.hide_polymer})",
        "hide_water": "state.viewerInstance.visual.visibility({water:!data.hide_water})",
        "hide_heteroatoms": "state.viewerInstance.visual.visibility({het:!data.hide_heteroatoms})",
        "hide_carbs": "state.viewerInstance.visual.visibility({carbs:!data.hide_carbs})",
        "hide_non_standard": "state.viewerInstance.visual.visibility({nonStandard:!data.hide_non_standard})",
        "hide_coarse": "state.viewerInstance.visual.visibility({coarse:!data.hide_coarse})",
        "hide_controls_icon": """self.rerender()""",
        "hide_expand_icon": """self.rerender()""",
        "hide_settings_icon": """self.rerender()""",
        "hide_selection_icon": """self.rerender()""",
        "hide_animation_icon": """self.rerender()""",
        "load_maps": "self.rerender()",
        "validation_annotation": """self.rerender()""",
        "domain_annotation": """self.rerender()""",
        "low_precision_coords": """self.rerender()""",
        "expanded": "state.viewerInstance.canvas.toggleExpanded(data.expanded)",
        "landscape": """self.rerender()""",
        "select_interaction": """self.rerender()""",
        "lighting": """self.rerender()""",
        "default_preset": """self.rerender()""",
        "pdbe_link": """self.rerender()""",
        "hide_controls": "state.viewerInstance.canvas.toggleControls(!data.hide_controls);",
        "spin": """state.viewerInstance.visual.toggleSpin(data.spin);""",
        "_select": """if(data._select) {state.viewerInstance.visual.select(data._select);}""",
        "_clear_selection": """state.viewerInstance.visual.clearSelection(data._args['number']);""",
        "_highlight": """if(data._highlight) {state.viewerInstance.visual.highlight(data._highlight);};""",
        "_clear_highlight": """state.viewerInstance.visual.clearHighlight();""",
        "_reset": """state.viewerInstance.visual.reset(data._args['data'])""",
        "resize": "state.viewerInstance.canvas.handleResize()",
    }

    def color(self, data, non_selected_color=None):
        self._select = {"data": data, "nonSelectedColor": non_selected_color}
        self._select = None

    def clear_selection(self, structure_number=None):
        self._args = {"number": structure_number}
        self._clear_selection = not self._clear_selection

    def highlight(self, data):
        self._highlight = {"data": data}
        self._highlight = None

    def clear_highlight(self):
        self._clear_highlight = not self._clear_highlight

    def reset(self, data):
        self._args = {"data": data}
        self._reset = not self._reset


# Functions
def value_to_mag_green(x, normalized=False):
    n = 1 if normalized else 255
    rgb_values = np.array([x, 1 - x, x]) * n
    return {ch: val for ch, val in zip(["r", "g", "b"], rgb_values)}


def value_to_color(x, normalized=False, cmap_func=cm.Reds):
    n = 1 if normalized else 255
    rgb_values = np.array(cmap_func(x)[:3]) * n
    return {ch: val for ch, val in zip(["r", "g", "b"], rgb_values)}


def normalize(x: np.ndarray):
    _min = np.min(x)
    _max = np.max(x)
    return (x - _min) / (_max - _min)


def find_numbered_labels(df):
    paratope_labels_mask = np.nonzero(df["full_paratope_labels"].iloc[0])[0]
    return np.array(df["antibody_imgt"].iloc[0])[paratope_labels_mask]


def find_numbered_preds(df):
    paratope_preds_mask = np.nonzero(df["predictions"].iloc[0])[0]
    return np.array(df["antibody_imgt"].iloc[0])[paratope_preds_mask]


def find_paratope_labels(chains, df):
    paratope_labels_selections = []

    for curr_chain in chains:
        tmp_df = df[df["chain_type"].isin([curr_chain])]
        chain_id_orig = tmp_df["antibody_chain"].iloc[0]
        paratope_labels_res_ins = find_numbered_labels(tmp_df)

        for res_ins in paratope_labels_res_ins:
            paratope_labels_selections.append({
                "struct_asym_id": str(chain_id_orig),
                "auth_residue_number": int(re.sub(r"\D", "", res_ins)),
                "auth_ins_code_id": str(re.sub(r"\d", "", res_ins)),
                "color": {"r": 255, "g": 20, "b": 0},
            })
    return paratope_labels_selections


def find_paratope_preds(chains, df):
    paratope_preds_selections = []

    for curr_chain in chains:
        tmp_df = df[df["chain_type"].isin([curr_chain])]
        chain_id_orig = tmp_df["antibody_chain"].iloc[0]
        paratope_preds_res_ins = find_numbered_preds(tmp_df)
        paratope_preds = tmp_df["predictions"].iloc[0]

        for res_ins, pred_prob in zip(paratope_preds_res_ins, paratope_preds):
            paratope_preds_selections.append({
                "struct_asym_id": str(chain_id_orig),
                "auth_residue_number": int(re.sub(r"\D", "", res_ins)),
                "auth_ins_code_id": str(re.sub(r"\d", "", res_ins)),
                "color": value_to_color(pred_prob),
            })

    return paratope_preds_selections


def find_models_diffs(chains, df_model1, df_model2, diff_mode):
    paratope_preds_diffs = []

    for curr_chain in chains:
        tmp_df1 = df_model1[df_model1["chain_type"] == curr_chain]
        tmp_df2 = df_model2[df_model2["chain_type"] == curr_chain]
        chain_id_orig = tmp_df1["antibody_chain"].iloc[0]
        paratope_preds_model1 = np.array(tmp_df1["predictions"].iloc[0])
        paratope_preds_model2 = np.array(tmp_df2["predictions"].iloc[0])

        if paratope_preds_model1.dtype == bool:
            paratope_preds_model1 = paratope_preds_model1.astype(int)
        if paratope_preds_model2.dtype == bool:
            paratope_preds_model2 = paratope_preds_model2.astype(int)

        # Normalized absolute probability difference.
        diff_models_preds = preds_diff(paratope_preds_model1, paratope_preds_model2, diff_mode)

        # Get the residue ids and insertion codes to update with a color.
        res_inss = tmp_df1["antibody_imgt"].to_list()[0]

        for res_ins, diff in zip(res_inss, diff_models_preds):
            paratope_preds_diffs.append({
                "struct_asym_id": str(chain_id_orig),
                "auth_residue_number": int(re.sub(r"\D", "", res_ins)),
                "auth_ins_code_id": str(re.sub(r"\d", "", res_ins)),
                "color": value_to_mag_green(diff)
            })
    return paratope_preds_diffs


def preds_diff(
        preds1: np.ndarray,
        preds2: np.ndarray,
        mode: str,
):
    if mode == "abs_norm_diff":
        diffs = np.abs(normalize(preds1) - normalize(preds2))
    elif mode == "abs_norm_rank_diff":
        diffs = np.abs(normalize(rankdata(preds1)) - normalize(rankdata(preds2)))
    else:
        raise ValueError("DifferenceMode not valid. You might have accidentally added a"
                         " new one without providing the implementation here")
    return np.clip(diffs, 0, 1)


@pn.cache(policy="LRU", max_items=5)
def load_pdbemolstar_pdb_pdb(pdb_id):
    global PDBE_MOLSTAR
    file_url = f"{ASSETS_NAME}/{pdb_id}.{STRUCTURE_EXT}"
    custom_data = {"url": file_url, "format": "cif", "binary": True}
    PDBE_MOLSTAR.param.update({"custom_data": custom_data})
    return PDBE_MOLSTAR


@pn.cache(policy="LRU", max_items=5)
def get_models_joined_df(model1, model2, chain, metric, region):
    df = DF[DF["metric"] == metric]
    df = df[df["region"] == region]

    parsed_chain = ["heavy", "light"] if chain == "both" else [chain]
    if len(parsed_chain) == 1:
        df = df[df["chain_type"] == parsed_chain[0]]

    model1_res = df[df["model"] == model1]
    model2_res = df[df["model"] == model2]

    model1 = f"{model1}_x"
    model2 = f"{model2}_y"

    on = ["pdb", "chain_type"]
    selection = ["pdb", "value", "value_right"]
    df = pd.merge(model1_res, model2_res, on=on, suffixes=("", "_right"), how="inner")[selection]
    df.columns = ["pdb", model1, model2]
    return df, model1, model2


@pn.depends(
    MODELS_X_WIDGET.param.value,
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
)
def scatter_plot(model1, model2, chain, metric, region):
    global SCATTER_PLOT_PANEL
    df, model1, model2 = get_models_joined_df(model1, model2, chain, metric, region)

    corr_coef = np.corrcoef(
        df[model1].to_numpy().flatten(),
        df[model2].to_numpy().flatten()
    )[0, 1]

    hover_data = [df[model1], df[model2], df["pdb"]]
    plot_title = f"{metric.upper()} -- Pearson Corr: {corr_coef:.2F}"

    sp = px.scatter(df, x=model1, y=model2, hover_data=hover_data)
    sp.update_layout(title_text=plot_title)
    sp.update_yaxes(range=[-0.1, 1.1])
    sp.update_xaxes(range=[-0.1, 1.1])
    sp.update_layout(PLOTLY_LAYOUT)
    sp.update_xaxes(PLOTLY_AXES_STYLE)
    sp.update_yaxes(PLOTLY_AXES_STYLE)
    SCATTER_PLOT_PANEL.object = sp
    return SCATTER_PLOT_PANEL


@pn.depends(
    MODELS_X_WIDGET.param.value,
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
)
def violin_plot(model1, model2, chain, metric, region):
    global VIOLIN_PLOT_PANEL
    parsed_chain = ["heavy", "light"] if chain == "both" else [chain]

    df = DF[DF["metric"].isin([metric])]
    df = df[df["region"].isin([region])]
    df = df[(df["model"].isin([model1, model2])) & (df["chain_type"].isin(parsed_chain))]
    df = df[["pdb", "model", "value"]]

    vp = px.violin(
        df,
        x="model",
        y="value",
        hover_data=[df["value"], df["pdb"]],
    )
    vp.update_layout(title_text=f"Distribution of {metric.upper()}")
    _ = [vio.update(span=[0, 1], spanmode='manual') for vio in vp.data]
    vp.update_layout(PLOTLY_LAYOUT)
    vp.update_xaxes(PLOTLY_AXES_STYLE)
    vp.update_yaxes(PLOTLY_AXES_STYLE)
    VIOLIN_PLOT_PANEL.object = vp
    return VIOLIN_PLOT_PANEL


@pn.depends(
    MODELS_X_WIDGET.param.value,
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
)
def radar_plot(model1, model2, chain, region):
    parsed_chain = ["heavy", "light"] if chain == "both" else [chain]

    df = DF[DF["region"] == region]
    df = df[df["model"].isin([model1, model2])]
    df = df[df["chain_type"].isin(parsed_chain)]
    df = df[["pdb", "model", "value", "metric"]]

    radar_current_metrics = df["metric"].unique()

    model1_df = df[df["model"] == model1]
    model2_df = df[df["model"] == model2]
    model1_means = [np.mean(model1_df[model1_df["metric"] == met]["value"]) for met in radar_current_metrics]
    model2_means = [np.mean(model2_df[model2_df["metric"] == met]["value"]) for met in radar_current_metrics]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=model1_means,
        theta=radar_current_metrics,
        fill="toself",
        name=model1,
    ))
    fig.add_trace(go.Scatterpolar(
        r=model2_means,
        theta=radar_current_metrics,
        fill="toself",
        name=model2,
    ))
    fig.update_layout(
        title_text=f"Comparison of selected models",
        legend=dict(
            yanchor="top",
            y=0.00,
            xanchor="left",
            x=0.01,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )),
        showlegend=True,
    )
    fig.update_layout(PLOTLY_LAYOUT)
    return pn.pane.Plotly(fig)


@pn.depends(
    MODELS_X_WIDGET.param.value,
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
)
def regions_plot(model1, model2, chain, metric):
    parsed_chain = ["heavy", "light"] if chain == "both" else [chain]

    df = DF[(DF["model"].isin([model1, model2])) & (DF["chain_type"].isin(parsed_chain)) & (DF["metric"] == metric)]
    df = df[["pdb", "model", "value", "region"]]

    fig = px.box(
        df,
        x="region",
        y="value",
        color="model",
    )
    fig.update_layout(title_text=f"Region performance with {metric.upper()}")
    fig.update_layout(PLOTLY_LAYOUT)
    fig.update_xaxes(PLOTLY_AXES_STYLE)
    fig.update_yaxes(PLOTLY_AXES_STYLE)
    return pn.pane.Plotly(fig)


@pn.depends(
    MODELS_X_WIDGET.param.value,
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
)
def mean_std_models(model1, model2, chain, metric, region):
    df, model1, model2 = get_models_joined_df(model1, model2, chain, metric, region)
    model1_mean = df[model1].mean()
    model2_mean = df[model2].mean()
    model1_std = df[model1].std()
    model2_std = df[model2].std()

    mean_std_model1 = f"{model1_mean:.2F} ± {model1_std:.2F}"
    mean_std_model2 = f"{model2_mean:.2F} ± {model2_std:.2F}"
    df = pd.DataFrame({
        "Selected models": [model1, model2],
        f"{metric.upper()} Mean ± Std": [mean_std_model1, mean_std_model2],
    })
    return pn.widgets.DataFrame(df, autosize_mode='fit_viewport', show_index=False)


# Updating the protein viewer for overlapped capabilities.
@pn.depends(
    MODELS_X_WIDGET.param.value,
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
    PDBS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
    DIFFERENCE_MODE_WIDGET.param.value,
)
def protein_viewer_overlap(*args):
    # Deepcopy is necessary since the pdbemolstar is cached,
    # i.e. when loaded to each viewer it must be unique s.t. controls work individually.
    prot_viewer = copy.deepcopy(load_pdbemolstar_pdb_pdb(args[4]))
    controller = pn.Param(ProteinParatopeOverlappedController(prot_viewer, DF, *args))
    extra_controller = pn.Param(prot_viewer, parameters=PDBE_EXTRA_PARAMS)

    gs = pn.GridSpec(sizing_mode="stretch_both")
    gs[0:1, 0:1] = pn.Column(controller, extra_controller, sizing_mode="stretch_both")
    gs[0:1, 1:4] = prot_viewer
    return gs


def protein_viewer_single(model, chain, metric, pdb_id, region):
    # Deepcopy is necessary since the pdbemolstar is cached,
    # i.e. when loaded to each viewer it must be unique s.t. controls work individually.
    prot_viewer = copy.deepcopy(load_pdbemolstar_pdb_pdb(pdb_id))
    controller_string = f"Model: {model} - Chain: {chain} - PDB ID: {pdb_id}"
    controller = pn.Param(
        ProteinParatopeSingleController(prot_viewer, DF, model, chain, metric, pdb_id, region),
        name=controller_string,
    )
    extra_controller = pn.Param(
        prot_viewer,
        parameters=PDBE_EXTRA_PARAMS,
    )
    return pn.GridBox(prot_viewer, controller, extra_controller, nrows=3, ncols=1)


# Since there are two different views we use these mock functions to reduce repeated code.
# SmallTODO: Find a better way to do this more elegantly.
# Updating the protein viewer for model on x-axis.
@pn.depends(
    MODELS_X_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
    PDBS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
)
def protein_viewer_x(*args):
    return protein_viewer_single(*args)


# Updating the protein viewer for model on y-axis.
@pn.depends(
    MODELS_Y_WIDGET.param.value,
    CHAINS_WIDGET.param.value,
    METRICS_WIDGET.param.value,
    PDBS_WIDGET.param.value,
    REGIONS_WIDGET.param.value,
)
def protein_viewer_y(*args):
    return protein_viewer_single(*args)


# Callbacks
def callback_click_pdb(target, event):
    target.value = event.new['points'][0]['customdata'][0]


def dashboard_app():
    """
        This dashboard allows you to explore antibody by antibody, residue by residue, model by model the predictions
        on a given result dataset on the paratope prediction task.
        The box plot with different regions, exposes the chosen metric from the Controls panel across the regions.
        The radar plot instead shows the performance of the chosen configuration across different preselected metrics.
        Two more plots help visualizing performance, the scatter plot highlights chain by chain how two selected models
        from the configuration perform. A linear relationship here indicates that the models behave similarly with
        respect to the chosen metric on the considered dataset. A violin plot helps visualize the distribution of the
        metric across antibody chains in the dataset. Finally, two tabs explore how the labels and the models behave
        with respect to one another, directly on the 3D representation of the protein.


        How does this work?

        - A file \`benchmark.pkl.gz\` encodes the results on a given dataset. This file is a Pandas DataFrame that
        contain the results. This can be created with a benchmarker from \`topefind.benchmarkers\` on a given
        train, validation, or test set.
        - There are several .bcif files that are served together, these need to match the structures that correspond to
        the PDB id exposed in the pdb column of \`benchmark.pkl.gz\`.
        - The app is serverless if converted to pyodide-worker, pyodide or pyscript. I suggest converting to
        pyodide-worker as it's the most stable or running it from python with panel.


        List of current issues and ugly fixes:

        - The layout is not adaptable to all screen sizes, especially for mobile, desktop is recommended.
        - Some Panel Rows grow a bit more than necessary for some screen sizes, if they don't render correctly zoom out.
        - The PDBeMolstar plugin does not fully expand sometimes (just trigger the controls button, and it will reset).
        - The PDBeMolstar has some troubles with the IMGT annotated structures, it might wrongly cartoon some residues
        other visual styles render correctly.

        If you find any other problems or solutions please open an issue under
        [topefind's issues](https://github.com/bayer-science-for-a-better-life/topefind/issues)
        """

    global SCATTER_PLOT_PANEL
    global PDBE_MOLSTAR

    PDBE_MOLSTAR = PDBeMolStar(
        sizing_mode="stretch_both",
        height=NORMAL_ROW_MAX_HEIGHT,
        custom_data={
            "url": f"{ASSETS_NAME}/{PREV_PDB}.{STRUCTURE_EXT}",
            "format": "cif",
            "binary": True
        }
    )
    # Linking necessary objects.
    SCATTER_PLOT_PANEL.link(PDBS_WIDGET, callbacks={"click_data": callback_click_pdb})

    # Set up the displayed texts.
    abstract_title = pn.pane.Markdown(
        "## Welcome to Topefind's dashboard 🚀",
        sizing_mode="stretch_width"
    )
    abstract = pn.pane.Markdown(dashboard_app.__doc__, sizing_mode="stretch_width")

    first_row_title = pn.pane.Markdown("#### Model comparisons", sizing_mode="stretch_width")
    first_row = pn.Row(regions_plot, radar_plot, sizing_mode="stretch_both")

    second_row_title = pn.pane.Markdown("#### Comparison w.r.t selected metric", sizing_mode="stretch_width")
    second_row = pn.Row(scatter_plot, violin_plot, sizing_mode="stretch_both")

    third_row_title = pn.pane.Markdown("#### Comparisons directly on the structure", sizing_mode="stretch_width")
    third_row = pn.Tabs(
        ("Overlapped", protein_viewer_overlap),
        ("Separate Views", pn.Row(protein_viewer_x, protein_viewer_y, sizing_mode="stretch_width")),
        dynamic=True
    )

    config_title = pn.pane.Markdown("### Controls ⚙️", sizing_mode="stretch_width")
    config_desc = pn.pane.Markdown(
        "Setting up the following widgets will expose several peculiarities of different methods for "
        "paratope prediction. "
        "The visualization focuses on the difference between two models at a time."
    )

    footer = pn.pane.Markdown("""
    Copyright (c) 2023, Bayer AG
    All rights reserved.
    """, sizing_mode="stretch_width")

    main = [
        pn.Card(
            abstract,
            header=abstract_title,
            background="WhiteSmoke",
            sizing_mode="stretch_both",
        ),
        pn.layout.Divider(),
        pn.Card(
            first_row,
            header=first_row_title,
            background="WhiteSmoke",
            sizing_mode="stretch_both",
        ),
        pn.layout.Divider(),
        pn.Card(
            second_row,
            header=second_row_title,
            background="WhiteSmoke",
            sizing_mode="stretch_both",
        ),
        pn.layout.Divider(),
        pn.Card(
            third_row,
            header=third_row_title,
            background="WhiteSmoke",
            sizing_mode="stretch_both",
        ),
        pn.layout.Divider(),
        footer,
    ]

    sidebar = [
        pn.Column(
            config_title,
            config_desc,
            MODELS_X_WIDGET,
            MODELS_Y_WIDGET,
            CHAINS_WIDGET,
            METRICS_WIDGET,
            REGIONS_WIDGET,
            PDBS_WIDGET,
            DIFFERENCE_MODE_WIDGET,
            pn.layout.Divider(),
            pn.pane.Markdown("Here is a summary with the given config:"),
            mean_std_models,
            sizing_mode="stretch_both"
        )
    ]
    # Dashboard App Layout.
    dashboard = pn.template.BootstrapTemplate(
        site="Dashboard",
        title="Topefind",
        main=main,
        sidebar=sidebar,
        sidebar_width=280,
        favicon=f"{ASSETS_NAME}/logo_only.ico"
    )
    # Serving or scripting everything
    if PYSCRIPT:
        dashboard.servable()
    else:
        pn.serve(
            dashboard,
            port=60001,
            static_dirs={
                ASSETS_NAME: Path(__file__).parent,
            },
        )


dashboard_app()


await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()