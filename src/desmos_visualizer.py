# =============================================================================
# desmos_visualizer.py
# Explainable AI Workbench — Desmos Visualization Engine
# =============================================================================

import re


def get_feature_bounds(df_clean, feature_name):
    """Return (min, max) of a feature from the cleaned dataset."""
    col = df_clean[feature_name].dropna()
    return float(col.min()), float(col.max())


def build_desmos_html(
    desmos_latex,
    x_feature,
    slider_configs,
    var_to_feature,
    fidelity_score=None,
    height=500
):
    """
    Build Desmos HTML with multiple sliders for all top features.

    Parameters:
        desmos_latex   : str  — equation in Desmos LaTeX (from sympy.latex())
        x_feature      : str  — name of the x-axis feature
        slider_configs : dict — {var_name: config_dict} e.g. {"a": {...}, "b": {...}}
                         config_dict keys: min, max, step, default, is_categorical, category_map
        var_to_feature : dict — {"x": "Present_Price", "a": "Year", ...}
        fidelity_score : float or None
        height         : int  — calculator height in pixels
    """

    # Escape equation for JavaScript
    eq_js = (
        desmos_latex
        .replace('\\', '\\\\')
        .replace("'", "\\'")
        .replace('"', '\\"')
    )

    # Build JavaScript expressions for all sliders
    slider_expressions = ""
    for var_name, cfg in slider_configs.items():
        feat_name = var_to_feature.get(var_name, var_name)
        default_val = cfg["default"]
        min_val = cfg["min"]
        max_val = cfg["max"]
        step_val = cfg["step"]

        slider_expressions += f"""
    calculator.setExpression({{
      id:    'slider_{var_name}',
      latex: '{var_name} = {default_val}',
      sliderBounds: {{ min: '{min_val}', max: '{max_val}', step: '{step_val}' }}
    }});"""

    # Build info bar labels
    info_labels = f"x = {x_feature}"
    for var_name, cfg in slider_configs.items():
        feat_name = var_to_feature.get(var_name, var_name)
        default_val = cfg["default"]
        if cfg.get("is_categorical") and cfg.get("category_map"):
            cat_label = cfg["category_map"].get(int(default_val), str(default_val))
            info_labels += f" | {var_name} = {feat_name}"
        else:
            info_labels += f" | {var_name} = {feat_name}"
    info_labels += " | y = Predicted Value"

    fidelity_badge = f"| Fidelity: {fidelity_score}%" if fidelity_score is not None else ""

    encoding_reference = "Encoding: "
    for var_name, cfg in slider_configs.items():
        if cfg.get("is_categorical") and cfg.get("category_map"):
            feat_name = var_to_feature.get(var_name, var_name)
            cats = " | ".join([f"{k}={v}" for k, v in cfg["category_map"].items()])
            encoding_reference += f"{var_name} ({feat_name}): {cats}  "

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:100%; height:{height}px; overflow:hidden; background:#ffffff; }}
  #dcalc {{ position:absolute; top:0; left:0; width:100%; height:{height}px; }}
</style>
</head>
<body>
  <div id="dcalc"></div>
  <script src="https://www.desmos.com/api/v1.9/calculator.js?apiKey=72ce9564c3264cd6b343beca4a21d5f6"></script>
  <script>
    window.addEventListener('load', function() {{
      var elt = document.getElementById('dcalc');
      var calculator = Desmos.GraphingCalculator(elt, {{
        backgroundColor: '#ffffff',
        border: false,
        expressionList: true,
        lockViewport: false,
        settingsMenu: false,
        zoomButtons: true,
        keypad: false,
        showGrid: true,
        showXAxis: true,
        showYAxis: true,
        xAxisNumbers: true,
        yAxisNumbers: true,
      }});

      calculator.resize();

      // Context note
      calculator.setExpression({{
        id: 'note1', type: 'text',
        text: '{info_labels} {fidelity_badge}'
      }});
      calculator.setExpression({{
        id: 'note2', type: 'text',
        text: '{encoding_reference}'
      }});

      // Sliders for all top features (except x-axis)
      {slider_expressions}

      // Surrogate equation
      calculator.setExpression({{
        id: 'surrogate',
        latex: 'y = {eq_js}',
        color: '#1565C0',
        lineWidth: 3
      }});

      setTimeout(function() {{ calculator.resize(); }}, 150);
    }});
  </script>
</body>
</html>"""

    return html