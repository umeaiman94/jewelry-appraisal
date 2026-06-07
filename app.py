import gradio as gr
from jewelry_model import JewelryAppraisal
from cohere_report import generate_provenance_report

# ── load / train model once ───────────────────────────────────────────────────
print("Initialising appraisal engine …")
appraiser = JewelryAppraisal.load_or_train()
print("Ready.\n")

# ── dropdown options ──────────────────────────────────────────────────────────
METALS     = ["Gold 18K", "Gold 14K", "Platinum", "Silver 925", "Rose Gold 18K"]
GEMSTONES  = ["None", "Diamond", "Ruby", "Emerald", "Sapphire", "Pearl"]
CUT_GRADES = ["N/A", "Excellent", "Very Good", "Good", "Fair"]
CONDITIONS = ["Mint", "Excellent", "Good", "Fair", "Poor"]

# ── core appraisal function ───────────────────────────────────────────────────
def appraise(metal_type, gemstone, carat_weight, cut_grade, condition):
    # validate
    if not metal_type:
        return "⚠ Please select a metal type.", ""
    if not condition:
        return "⚠ Please select a condition.", ""

    # auto-correct cut / carat for no-gemstone pieces
    if gemstone == "None" or gemstone == "":
        carat_weight = 0.0
        cut_grade    = "N/A"
    else:
        try:
            carat_weight = float(carat_weight)
            if carat_weight <= 0:
                carat_weight = 0.0
                gemstone     = "None"
                cut_grade    = "N/A"
        except (ValueError, TypeError):
            carat_weight = 0.0
            gemstone     = "None"
            cut_grade    = "N/A"

    # KNN prediction
    price = appraiser.predict(metal_type, gemstone, carat_weight, cut_grade, condition)
    price_display = f"**Estimated Fair Market Value: USD {price:,.2f}**"

    # Cohere provenance report
    report = generate_provenance_report(
        metal_type, gemstone, carat_weight, cut_grade, condition, price
    )

    return price_display, report


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
body { font-family: 'Georgia', serif; background: #0f0e0b; color: #e8dcc8; }
.gradio-container { max-width: 900px; margin: auto; }
#title { text-align: center; font-size: 2rem; letter-spacing: 0.15em;
         color: #c9a96e; margin-bottom: 0.25rem; }
#subtitle { text-align: center; color: #8a7a60; font-style: italic;
            margin-bottom: 2rem; }
.label { color: #c9a96e !important; font-weight: bold; }
#appraise-btn { background: #c9a96e; color: #0f0e0b; font-weight: bold;
                font-size: 1rem; border-radius: 4px; padding: 10px 28px; }
#appraise-btn:hover { background: #e4c48a; }
#price-out { font-size: 1.4rem; text-align: center; color: #c9a96e;
             border: 1px solid #3a3020; border-radius: 6px; padding: 14px;
             background: #1a1810; }
#report-out { background: #141210; border: 1px solid #3a3020;
              border-radius: 6px; padding: 16px; line-height: 1.75;
              color: #d4c4a0; font-size: 0.95rem; }
"""

with gr.Blocks(css=css, title="Luxury Jewelry Appraisal") as demo:

    gr.HTML('<div id="title">◈ PRESTIGE APPRAISAL</div>')
    gr.HTML('<div id="subtitle">AI-Powered Fair Market Valuation for Pre-Owned Jewelry</div>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Jewelry Specifications")

            metal_type = gr.Dropdown(
                choices=METALS, label="Metal Type", elem_classes="label",
                info="Alloy and purity of the primary metal"
            )
            gemstone = gr.Dropdown(
                choices=GEMSTONES, value="None", label="Gemstone",
                elem_classes="label", info="Select 'None' for metal-only pieces"
            )
            carat_weight = gr.Number(
                value=0.0, label="Carat Weight", minimum=0.0, maximum=20.0,
                step=0.01, elem_classes="label",
                info="Total gemstone weight in carats (0 if no gemstone)"
            )
            cut_grade = gr.Dropdown(
                choices=CUT_GRADES, value="N/A", label="Cut Grade",
                elem_classes="label",
                info="GIA cut grade — select N/A for pearls or no gemstone"
            )
            condition = gr.Dropdown(
                choices=CONDITIONS, label="Piece Condition", elem_classes="label",
                info="Overall physical condition of the piece"
            )

            appraise_btn = gr.Button("Run Appraisal", elem_id="appraise-btn",
                                      variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Appraisal Results")
            price_out  = gr.Markdown(elem_id="price-out",
                                      value="_Price will appear here after appraisal_")
            report_out = gr.Markdown(elem_id="report-out",
                                      value="_Provenance report will appear here after appraisal_")

    appraise_btn.click(
        fn=appraise,
        inputs=[metal_type, gemstone, carat_weight, cut_grade, condition],
        outputs=[price_out, report_out],
    )

    gr.HTML("""
    <div style="text-align:center; color:#5a4f3a; font-size:0.78rem; margin-top:2rem;">
        Valuations are estimates generated by a K-Nearest Neighbours regression model
        trained on secondary-market data. Not a substitute for professional gemological appraisal.
    </div>""")

if __name__ == "__main__":
    demo.launch(share=False)
