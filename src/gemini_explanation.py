def generate_explanation(
    api_key,
    selected_model,
    problem_type,
    best_equation,
    fidelity_score,
    top_features,
    importance_df,
    x_feature,
    slider_features,
    active_slider_features,
    anchored_values,
    encoding_maps=None
):
    from google import genai

    client = genai.Client(api_key=api_key)

    importance_text = ""
    for _, row in importance_df.iterrows():
        importance_text += f"  - {row['feature']}: {row['importance_pct']:.1f}%\n"

    omitted_features = [f for f in slider_features if f not in active_slider_features]

    prompt = f"""You are an intuitive, grounded AI Explainer for an Explainable AI Workbench.

STRICT ACCURACY CONSTRAINTS (DO NOT VIOLATE):
1. Primary Feature (X-Axis): `{x_feature}`. It is read directly from the graph plot. NEVER suggest dragging a slider for `{x_feature}`.
2. ACTIVE SLIDERS IN MATH FORMULA: {active_slider_features if active_slider_features else 'None'}.
   - ONLY suggest interactive experiments using active sliders: {active_slider_features}.
3. OMITTED / DROPPED FEATURES (NOT IN MATH FORMULA): {omitted_features if omitted_features else 'None'}.
   - You MUST explicitly state in Section 4 that PySR OMITTED/DROPPED these features ({omitted_features}) from the formula to simplify the math.
   - NEVER tell the user to adjust sliders for omitted features ({omitted_features}) because they DO NOT exist in the equation!

CONTEXT:
- ML Model: {selected_model} ({problem_type})
- Primary Feature (X-Axis): {x_feature}
- Active Sliders: {active_slider_features}
- Omitted Features: {omitted_features}
- Surrogate Equation: f = {best_equation}
- Fidelity Score: {fidelity_score}%
- Feature Importances: 
{importance_text}

INSTRUCTIONS FOR EXPLANATION:
• 1. The Big Picture
Explain how {x_feature} drives predictions with a simple real-world analogy.

• 2. Key Mathematical Intuition
Translate the math formula `f = {best_equation}` into plain English.

• 3. Interactive Experiment Suggestion
Suggest an experiment using ONLY active sliders: {active_slider_features if active_slider_features else 'None'}. Explain how adjusting them alters the graph. If there are no active sliders, inform the user that the curve is fixed.

• 4. Fidelity & Exclusions
Highlight the {fidelity_score}% fidelity score. Mention that PySR omitted/dropped these features from the math equation due to minimal impact: {omitted_features if omitted_features else 'None'}. Note that adjusting their sliders on screen will have no effect on the graph curve since they do not appear in the math.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text