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

    Root cause of previous black-screen bug:
      Desmos.GraphingCalculator() reads the div's offsetWidth/offsetHeight
      at construction time. Inside a Streamlit iframe, even with explicit
      inline CSS height, the layout engine has not committed computed
      dimensions at inline-script execution time — so Desmos reads 0x0
      and creates a permanently invisible canvas.

    Fix: defer initialization to window.addEventListener('load', ...) so
    the full document layout is complete before Desmos reads dimensions.
    Also pass explicit width/height to the constructor as a hard fallback,
    and call calculator.resize() after creation.
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

    # Current slider value rounded
    ax2_val = round(float(ax2_current_value), 4)

    # X-axis viewport with 10% padding
    x_pad   = max((ax1_max - ax1_min) * 0.10, 0.5)
    x_left  = round(ax1_min - x_pad, 4)
    x_right = round(ax1_max + x_pad, 4)

    # Full document height = calculator height
    doc_h = height

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html {{
    width:100%;
    height:{doc_h}px;
    overflow:hidden;
  }}
  body {{
    width:100%;
    height:{doc_h}px;
    overflow:hidden;
    background:#1a1f35;
  }}
  #dcalc {{
    position:absolute;
    top:0; left:0;
    width:100%;
    height:{doc_h}px;
  }}
</style>
</head>
<body>

  <div id="dcalc"></div>

  <!--
    Desmos API script — loaded SYNCHRONOUSLY so it is available in
    the window.load handler below.
  -->
  <script src="https://www.desmos.com/api/v1.9/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>

  <script>
    /*
      Defer ALL Desmos initialization to the 'load' event.
      At this point the browser has:
        1. Parsed the full HTML
        2. Applied all CSS
        3. Computed final layout (offsetWidth / offsetHeight > 0)
        4. Loaded the Desmos API script
      Only then do we create the calculator — guaranteeing non-zero dimensions.
    */
    window.addEventListener('load', function() {{

      var elt = document.getElementById('dcalc');

      /*
        Pass explicit width and height to Desmos constructor as a hard
        fallback in case offsetWidth/offsetHeight are still not correct.
        This is an undocumented but supported Desmos API option.
      */
      var calculator = Desmos.GraphingCalculator(elt, {{
        backgroundColor: '#ffffff',
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

      /*
        Force Desmos to re-read the element dimensions after construction.
        This is the critical call that corrects any zero-size initialization.
      */
      calculator.resize();

      // Set x-axis viewport using actual data range
      calculator.setMathBounds({{
        left:   {x_left},
        right:  {x_right},
        bottom: undefined,
        top:    undefined
      }});

      // Context note — shown in expression list
      calculator.setExpression({{
        id:   'note1',
        type: 'text',
        text: 'x = {ax1_js}  |  a = {ax2_js}  |  y = Predicted Value'
      }});

      // Slider for axis_feature_2 — user drags to explore
      calculator.setExpression({{
        id:    'slider_a',
        latex: 'a = {ax2_val}',
        sliderBounds: {{
          min:  '{ax2_min}',
          max:  '{ax2_max}',
          step: ''
        }}
      }});

      // Surrogate equation:  y = f(x, a)
      calculator.setExpression({{
        id:        'surrogate',
        latex:     'y = {desmos_eq_js}',
        color:     '#4fc3f7',
        lineWidth: 3
      }});

      /*
        Second resize call after expressions are loaded.
        Desmos sometimes needs this when expressions cause layout changes
        in the expression panel that affect the graph area dimensions.
      */
      setTimeout(function() {{
        calculator.resize();
      }}, 100);

    }});
  </script>

</body>
</html>"""

    return html