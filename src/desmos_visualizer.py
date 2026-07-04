# =============================================================================
# desmos_visualizer.py
# Explainable AI Workbench — Desmos Visualization Engine
# =============================================================================

import re


def format_equation_for_desmos(desmos_equation_str):
    """
    Convert equation string (x and y as variables) to Desmos-compatible format.
    - y → a  (axis_feature_2 becomes slider 'a'; y is Desmos output axis)
    - log → ln  (PySR uses natural log; Desmos log = base-10)
    - ** → ^  (safety pass)
    """
    eq = desmos_equation_str
    eq = re.sub(r'\by\b', 'a', eq)
    eq = re.sub(r'\blog\b', 'ln', eq)
    eq = eq.replace('**', '^')
    return eq


def get_feature_bounds(df_clean, feature_name):
    """Return (min, max) of a feature from the cleaned dataset."""
    col = df_clean[feature_name].dropna()
    return float(col.min()), float(col.max())


def build_desmos_html(
    desmos_equation,
    axis_feature_1,
    axis_feature_2,
    ax2_current_value,
    ax2_min,
    ax2_max,
    ax1_min,
    ax1_max,
    fidelity_score=None,
    height=500
):
    """
    Build a self-contained HTML page with a Desmos GraphingCalculator.

    The graph plots:  y = f(x, a)
      x  = axis_feature_1   (horizontal axis, free variable)
      a  = axis_feature_2   (Desmos slider — user drags to explore)
      y  = model prediction (vertical axis)

    Critical design decisions for reliable rendering:
      1. Script tag is AFTER the div (ensures div exists when Desmos initializes)
      2. Inline style on the div with explicit px height (Desmos reads this)
      3. html/body set to same height with overflow:hidden (no scrollbars)
      4. No wrapper divs (nothing to interfere with height computation)
    """

    # Convert equation to Desmos format
    desmos_eq = format_equation_for_desmos(desmos_equation)

    # Escape for safe JavaScript string embedding
    desmos_eq_js = (
        desmos_eq
        .replace('\\', '\\\\')
        .replace("'",  "\\'")
        .replace('"',  '\\"')
    )

    # Escape feature names for JavaScript
    ax1_js = axis_feature_1.replace("'", "\\'")
    ax2_js = axis_feature_2.replace("'", "\\'")

    # Current slider value
    ax2_val = round(float(ax2_current_value), 4)

    # X-axis viewport with 10% padding
    x_pad   = max((ax1_max - ax1_min) * 0.10, 0.5)
    x_left  = round(ax1_min - x_pad, 4)
    x_right = round(ax1_max + x_pad, 4)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{
    width:100%;
    height:{height}px;
    overflow:hidden;
    background:#111827;
  }}
</style>
</head>
<body>

  <!-- Calculator div BEFORE the script — Desmos needs it to exist first -->
  <div id="dcalc" style="width:100%; height:{height}px;"></div>

  <!-- Desmos API loaded synchronously AFTER the div -->
  <script src="https://www.desmos.com/api/v1.9/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0faa6"></script>

  <!-- Initialize immediately after API loads -->
  <script>
    var elt = document.getElementById('dcalc');

    var calculator = Desmos.GraphingCalculator(elt, {{
      backgroundColor: '#111827',
      border:          false,
      expressionList:  true,
      lockViewport:    false,
      settingsMenu:    false,
      zoomButtons:     true,
      keypad:          false,
      showGrid:        true,
      showXAxis:       true,
      showYAxis:       true,
      xAxisNumbers:    true,
      yAxisNumbers:    true,
    }});

    // Set x-axis viewport
    calculator.setMathBounds({{
      left:   {x_left},
      right:  {x_right},
      bottom: undefined,
      top:    undefined
    }});

    // Note explaining the variable mapping
    calculator.setExpression({{
      id:   'note1',
      type: 'text',
      text: 'x = {ax1_js}  |  a = {ax2_js}  |  y = Predicted Value'
    }});

    // Slider for axis_feature_2
    // User drags this to instantly update the curve
    calculator.setExpression({{
      id:    'slider_a',
      latex: 'a = {ax2_val}',
      sliderBounds: {{
        min:  '{ax2_min}',
        max:  '{ax2_max}',
        step: ''
      }}
    }});

    // The surrogate equation:  y = f(x, a)
    calculator.setExpression({{
      id:        'surrogate',
      latex:     'y = {desmos_eq_js}',
      color:     '#4fc3f7',
      lineWidth: 3
    }});

  </script>
</body>
</html>"""

    return html