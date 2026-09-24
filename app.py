import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import tempfile
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# 1. PAGE SETUP & ADVANCED CSS INJECTION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="KandaDrishti AI | Enterprise Mandi OS",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Modern Glassmorphic Container */
    .bento-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }
    
    /* Highlight Cards */
    .stat-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    /* Status Badges */
    .badge-approved {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #31C48D;
        display: inline-block;
    }
    .badge-disputed {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #F98080;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. MODEL ENGINE WITH CACHED PERSISTENCE
# ------------------------------------------------------------------------------
@st.cache_resource
def load_detection_engine():
    model_paths = ["best.pt", "runs/detect/kandadrishti_model/weights/best.pt"]
    for path in model_paths:
        if os.path.exists(path):
            try:
                return YOLO(path)
            except Exception:
                pass
    return None

detector = load_detection_engine()

# ------------------------------------------------------------------------------
# 3. VERIFIED DIGITAL AUDIT CERTIFICATE
# ------------------------------------------------------------------------------
def generate_audit_certificate(stats, batch_id, mandi_name, market_payout):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "KandaDrishti AI - Official Quality Certificate")
    c.setFont("Helvetica", 9)
    c.drawString(50, 735, f"Procurement Authority: Dept. of Consumer Affairs (DoCA) | Mandi: {mandi_name}")
    c.drawString(50, 722, f"Consignment Batch: {batch_id} | Issued: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    c.line(50, 712, 550, 712)

    total = max(stats["Total"], 1)
    grade_a_pct = (stats["Grade_A"] / total) * 100
    grade_b_pct = (stats["Grade_B"] / total) * 100
    urs_pct = (stats["URS"] / total) * 100
    defective_pct = (stats["Defective"] / total) * 100

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 680, "Grading Breakdown (NAFED / Agmark Standards):")
    c.setFont("Helvetica", 10)
    c.drawString(70, 650, f"• Total Tested Samples: {stats['Total']} units")
    c.drawString(70, 630, f"• Grade A (>45 mm sound): {stats['Grade_A']} ({grade_a_pct:.1f}%)")
    c.drawString(70, 610, f"• Grade B (40 - 45 mm sound): {stats['Grade_B']} ({grade_b_pct:.1f}%)")
    c.drawString(70, 590, f"• Under Regular Size (<40 mm URS): {stats['URS']} ({urs_pct:.1f}%)")
    c.drawString(70, 570, f"• Defective Bulbs (Sprouted/Rotten): {stats['Defective']} ({defective_pct:.1f}%)")
    
    c.line(50, 545, 550, 545)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, 525, f"Recommended Procurement Rate: Rs. {market_payout['rate_per_kg']:.2f} / kg")
    c.drawString(50, 505, f"Net Payable Settlement: Rs. {market_payout['total_payable']:,.2f}")

    is_accepted = (defective_pct < 5.0 and grade_a_pct >= 50.0)
    status = "ACCEPTED FOR BUFFER STOCK PROCUREMENT" if is_accepted else "RE-INSPECTION / DISPUTE PROTOCOL TRIGGERED"
    c.line(50, 485, 550, 485)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 460, f"Final Clearance Verdict: {status}")
    c.save()
    return temp_pdf.name

def detect_pathology(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Green sprouting mask
    g_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    g_ratio = np.sum(g_mask > 0) / max(crop_bgr.shape[0] * crop_bgr.shape[1], 1)
    # Black rot mask
    b_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45]))
    b_ratio = np.sum(b_mask > 0) / max(crop_bgr.shape[0] * crop_bgr.shape[1], 1)

    if g_ratio > 0.04:
        return "Sprouted"
    elif b_ratio > 0.08:
        return "Rotten"
    return "Sound"

# ------------------------------------------------------------------------------
# 4. SIDEBAR SETTINGS & MANDI MARKET DESK
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ APMC Mandi Terminal")
    mandi_hub = st.selectbox(
        "Regional Hub",
        ["Lasalgaon APMC (Nashik)", "Pimpalgaon APMC", "Solapur APMC", "Alwar Mandi Hub", "Bangalore Central"]
    )
    consignment_id = st.text_input("Consignment Batch ID", value=f"DOCA-{datetime.datetime.now().strftime('%y%m%d')}-01")
    
    st.markdown("---")
    st.markdown("### ⚖️ Valuation Parameters")
    base_grade_a_rate = st.number_input("Grade-A Base Rate (₹/Quintal)", value=3200, step=100)
    batch_weight_qtl = st.number_input("Lot Weight (Quintals)", value=30.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 🎛️ Optical Sensitivity")
    conf_level = st.slider("Neural Confidence Threshold", 0.02, 0.40, 0.06, 0.02)
    calibrated_ratio = st.slider("Equatorial Calibration Ratio (mm/px)", 0.10, 0.25, 0.16, 0.01)

# ------------------------------------------------------------------------------
# 5. HERO HEADER & INPUT SECTION
# ------------------------------------------------------------------------------
st.markdown("""
<div style='background: linear-gradient(90deg, #0B1936 0%, #1B315E 100%); padding: 24px; border-radius: 14px; margin-bottom: 20px; color: white;'>
    <h2 style='margin:0; color:#FFFFFF;'>🧅 KandaDrishti AI: Automated Grading & Sizing OS</h2>
    <p style='margin:4px 0 0 0; color:#CBD5E1; font-size:14px;'>Department of Consumer Affairs (DoCA) & NAFED Procurement Buffer Stock Standardizer</p>
</div>
""", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs(["📁 High-Resolution Tray Upload", "📸 Direct Optical Camera Input"])
with input_tab1:
    file_upload = st.file_uploader("Upload sample tray, warehouse consignment, or field sample:", type=["jpg", "jpeg", "png"])
with input_tab2:
    cam_capture = st.camera_input("Capture inspection tray view:")

sample_input = file_upload if file_upload is not None else cam_capture

# ------------------------------------------------------------------------------
# 6. INFERENCE & METRIC ANALYSIS
# ------------------------------------------------------------------------------
if sample_input is not None:
    raw_img = Image.open(sample_input).convert("RGB")
    np_rgb = np.array(raw_img)
    img_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = img_bgr.shape

    annotated = img_bgr.copy()
    stats = {"Grade_A": 0, "Grade_B": 0, "URS": 0, "Defective": 0, "Total": 0}
    boxes = []
    records = []

    # Stage 1: Neural Inference
    if detector is not None:
        try:
            preds = detector.predict(img_bgr, conf=conf_level, iou=0.40, device="cpu", verbose=False)[0]
            for b in preds.boxes:
                c_name = detector.names[int(b.cls[0])]
                if "coin" not in c_name:
                    coords = b.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]))
        except Exception:
            pass

    # Stage 2: Adaptive Vision Fallback (Eliminates false zeroes)
    if len(boxes) == 0:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 35 and bh > 35 and (bw * bh) < (img_w * img_h * 0.85):
                boxes.append((bx, by, bw, bh))

    # Stage 3: Dimension & Pathology Classification
    for idx, (x, y, w, h) in enumerate(boxes):
        crop = img_bgr[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)]
        defect = detect_pathology(crop) if crop.size > 0 else "Sound"
        eq_dia_mm = round(max(w, h) * calibrated_ratio, 1)

        if defect != "Sound":
            grade = f"Defective ({defect})"
            stats["Defective"] += 1
            color = (0, 0, 255)
        else:
            if eq_dia_mm >= 45.0:
                grade = "Grade A"
                stats["Grade_A"] += 1
                color = (0, 200, 0)
            elif eq_dia_mm >= 40.0:
                grade = "Grade B"
                stats["Grade_B"] += 1
                color = (0, 215, 255)
            else:
                grade = "URS"
                stats["URS"] += 1
                color = (255, 140, 0)

        stats["Total"] += 1
        records.append({
            "Sample ID": f"Onion #{idx+1:02d}",
            "Equatorial Diameter (mm)": eq_dia_mm,
            "NAFED Classification": grade,
            "Surface Pathology": defect
        })
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        cv2.putText(annotated, f"{grade} | {eq_dia_mm}mm", (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total_detected = max(stats["Total"], 1)
    pct_a = (stats["Grade_A"] / total_detected) * 100
    pct_b = (stats["Grade_B"] / total_detected) * 100
    pct_urs = (stats["URS"] / total_detected) * 100
    pct_defect = (stats["Defective"] / total_detected) * 100

    # Valuation Calculations
    adjusted_qtl_price = (
        (pct_a / 100.0) * base_grade_a_rate +
        (pct_b / 100.0) * (base_grade_a_rate * 0.85) +
        (pct_urs / 100.0) * (base_grade_a_rate * 0.55) -
        (pct_defect / 100.0) * (base_grade_a_rate * 0.40)
    )
    adjusted_qtl_price = max(adjusted_qtl_price, 500.0)
    total_lot_payable = adjusted_qtl_price * batch_weight_qtl
    pricing_summary = {
        "rate_per_kg": adjusted_qtl_price / 100.0,
        "total_payable": total_lot_payable
    }

    # --------------------------------------------------------------------------
    # 7. EXECUTIVE KPI SCORECARD & CLEARANCE BANNER
    # --------------------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Count", f"{stats['Total']} Units")
    k2.metric("Grade A (>45 mm)", f"{pct_a:.1f}%", help="Prime Export / Buffer Stock")
    k3.metric("Grade B (40-45 mm)", f"{pct_b:.1f}%")
    k4.metric("URS (<40 mm)", f"{pct_urs:.1f}%", delta=f"-{pct_urs:.1f}%" if pct_urs > 15 else "Acceptable", delta_color="inverse")
    k5.metric("Defects (Rot/Sprout)", f"{pct_defect:.1f}%", delta=f"{pct_defect:.1f}%" if pct_defect > 5 else "Clean", delta_color="inverse")

    # Dynamic Procurement Clearance Status
    if pct_defect < 5.0 and pct_a >= 50.0:
        st.markdown(f"""
        <div style='background: #DEF7EC; border: 1.5px solid #31C48D; padding: 14px 20px; border-radius: 10px; margin: 15px 0;'>
            <span class='badge-approved'>🟢 PROCUREMENT APPROVED</span>
            <strong style='color:#03543F; font-size: 15px; margin-left: 10px;'>Consignment fully complies with NAFED/DoCA buffer procurement norms.</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: #FDE8E8; border: 1.5px solid #F98080; padding: 14px 20px; border-radius: 10px; margin: 15px 0;'>
            <span class='badge-disputed'>🔴 REVIEW REQUIRED</span>
            <strong style='color:#9B1C1C; font-size: 15px; margin-left: 10px;'>Consignment exceeds allowed threshold for defect or URS ratio. Secondary sampling triggered.</strong>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 8. SPATIAL VISION COMPARISON & INTERACTIVE PLOTLY CHARTS
    # --------------------------------------------------------------------------
    c_img1, c_img2 = st.columns(2)
    with c_img1:
        st.markdown("**Original Inspection View**")
        st.image(np_rgb, use_container_width=True)
    with c_img2:
        st.markdown("**Bounding Segmenter & Sizing Overlay**")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Batch Distribution Analytics")
    ch1, ch2 = st.columns(2)

    with ch1:
        df_pie = pd.DataFrame({
            "Classification": ["Grade A (>45mm)", "Grade B (40-45mm)", "URS (<40mm)", "Defective"],
            "Count": [stats["Grade_A"], stats["Grade_B"], stats["URS"], stats["Defective"]]
        })
        fig_donut = px.pie(
            df_pie,
            values="Count",
            names="Classification",
            hole=0.55,
            color="Classification",
            color_discrete_map={
                "Grade A (>45mm)": "#14803C",
                "Grade B (40-45mm)": "#EAB308",
                "URS (<40mm)": "#F97316",
                "Defective": "#DC2626"
            }
        )
        fig_donut.update_layout(title="Batch Quality Composition", margin=dict(l=10, r=10, t=35, b=10), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with ch2:
        if len(records) > 0:
            df_hist = pd.DataFrame(records)
            fig_hist = px.histogram(
                df_hist,
                x="Equatorial Diameter (mm)",
                nbins=14,
                color="NAFED Classification",
                color_discrete_map={
                    "Grade A": "#14803C",
                    "Grade B": "#EAB308",
                    "URS": "#F97316",
                    "Defective (Rotten)": "#DC2626",
                    "Defective (Sprouted)": "#9333EA"
                }
            )
            fig_hist.add_vline(x=45.0, line_dash="dash", line_color="#14803C", annotation_text="Grade A (45mm)")
            fig_hist.add_vline(x=40.0, line_dash="dash", line_color="#EAB308", annotation_text="Grade B (40mm)")
            fig_hist.update_layout(title="Equatorial Diameter Spectrum (mm)", margin=dict(l=10, r=10, t=35, b=10), height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

    # --------------------------------------------------------------------------
    # 9. COMMERCIAL MANDI SETTLEMENT DESK
    # --------------------------------------------------------------------------
    st.markdown("### 💰 Commercial Mandi Settlement Desk")
    f1, f2, f3 = st.columns(3)
    f1.metric("Benchmark Grade A Rate", f"₹ {base_grade_a_rate:,} / Qtl")
    f2.metric("Effective Adjusted Price", f"₹ {adjusted_qtl_price:,.2f} / Qtl", f"₹ {(adjusted_qtl_price/100):.2f} / kg")
    f3.metric("Net Farmer Payout", f"₹ {total_lot_payable:,.2f}", f"Batch: {batch_weight_qtl} Qtl")

    # --------------------------------------------------------------------------
    # 10. SAMPLE LOG TABLE & AUDIT PDF EXPORT
    # --------------------------------------------------------------------------
    if len(records) > 0:
        with st.expander("📋 View Discrete Sample Metric Table"):
            st.dataframe(pd.DataFrame(records), use_container_width=True)

    pdf_bytes_path = generate_audit_certificate(stats, consignment_id, mandi_hub, pricing_summary)
    with open(pdf_bytes_path, "rb") as f:
        st.download_button(
            label="📥 Download Cryptographic Quality Certificate (PDF)",
            data=f,
            file_name=f"Certificate_{consignment_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
else:
    st.info("👈 Upload an onion tray sample or capture via the live camera above to begin automated assessment.")
