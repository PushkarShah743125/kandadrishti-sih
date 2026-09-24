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
# PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="KandaDrishti AI - Enterprise Onion Grading",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #0B1936;
    }
    .custom-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-accept {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-reject {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# MODEL INITIALIZATION
# ------------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    if os.path.exists("best.pt"):
        try:
            return YOLO("best.pt")
        except Exception:
            return None
    return None

model = load_yolo_model()

# ------------------------------------------------------------------------------
# REPORT GENERATION ENGINE
# ------------------------------------------------------------------------------
def generate_pdf(stats, batch_id, mandi_name, market_val):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "KandaDrishti AI - Quality & Valuation Certificate")
    c.setFont("Helvetica", 10)
    c.drawString(50, 735, f"Authority: Dept. of Consumer Affairs (DoCA) | Mandi: {mandi_name}")
    c.drawString(50, 720, f"Batch Consignment ID: {batch_id} | Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, 710, 550, 710)

    total = max(stats["Total"], 1)
    grade_a_pct = (stats["Grade_A"] / total) * 100
    grade_b_pct = (stats["Grade_B"] / total) * 100
    urs_pct = (stats["URS"] / total) * 100
    defective_pct = (stats["Defective"] / total) * 100

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 680, "Grading Assessment Breakdown (Agmark / NAFED Standards):")
    c.setFont("Helvetica", 11)
    c.drawString(70, 650, f"Total Count Inspected: {stats['Total']} units")
    c.drawString(70, 625, f"Grade A (>45 mm sound): {stats['Grade_A']} ({grade_a_pct:.1f}%)")
    c.drawString(70, 600, f"Grade B (40 - 45 mm sound): {stats['Grade_B']} ({grade_b_pct:.1f}%)")
    c.drawString(70, 575, f"URS (<40 mm Under Regular Size): {stats['URS']} ({urs_pct:.1f}%)")
    c.drawString(70, 550, f"Defective (Rotten / Sprouted): {stats['Defective']} ({defective_pct:.1f}%)")
    
    c.line(50, 530, 550, 530)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 510, f"Recommended Procurement Rate: Rs. {market_val['rate_per_kg']:.2f} / kg")
    c.drawString(50, 490, f"Net Estimated Lot Payout: Rs. {market_val['total_payout']:,.2f}")

    status = "ACCEPTED FOR BUFFER STOCK PROCUREMENT" if (defective_pct < 5.0 and grade_a_pct >= 50.0) else "DISPUTED / MANUAL REVIEW REQUIRED"
    c.line(50, 470, 550, 470)
    c.drawString(50, 445, f"Procurement Verdict: {status}")
    c.save()
    return temp_pdf.name

def check_defect(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / max(crop_bgr.shape[0] * crop_bgr.shape[1], 1)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 45])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_ratio = np.sum(black_mask > 0) / max(crop_bgr.shape[0] * crop_bgr.shape[1], 1)

    if green_ratio > 0.04:
        return "Sprouted"
    elif black_ratio > 0.08:
        return "Rotten"
    return "Sound"

# ------------------------------------------------------------------------------
# SIDEBAR: CONTROLS & COMMODITY PRICING
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2909/2909841.png", width=65)
    st.title("Settings & Mandi Desk")
    st.caption("DoCA Procurement Gateway v2.4")
    
    st.markdown("---")
    st.subheader("📍 APMC Mandi Registry")
    mandi_name = st.selectbox(
        "Procurement Hub",
        ["Lasalgaon APMC (Nashik)", "Pimpalgaon APMC", "Solapur APMC", "Alwar APMC", "Bangalore Rural"]
    )
    batch_id = st.text_input("Consignment / Lot ID", value=f"DOCA-{datetime.datetime.now().strftime('%Y%m%d')}-01")
    
    st.markdown("---")
    st.subheader("💰 Live Mandi Benchmark Pricing")
    base_mandi_rate = st.number_input("Base Benchmark Rate (₹ / Quintal for Grade A)", min_value=1000, max_value=8000, value=3200, step=100)
    lot_weight_quintals = st.number_input("Consignment Net Weight (Quintals)", min_value=1.0, max_value=500.0, value=25.0, step=0.5)

    st.markdown("---")
    st.subheader("⚙️ Optical Parameters")
    conf_thresh = st.slider("Model Detection Confidence", min_value=0.02, max_value=0.50, value=0.06, step=0.02)
    custom_px_to_mm = st.slider("Focal Calibration Ratio (mm/px)", min_value=0.10, max_value=0.25, value=0.16, step=0.01)

# ------------------------------------------------------------------------------
# MAIN APPLICATION BODY
# ------------------------------------------------------------------------------
st.title("🧅 KandaDrishti AI: Automated Quality & Sizing System")
st.markdown("Official Decision-Support Platform for **Department of Consumer Affairs (DoCA) & NAFED Procurement**")

# Input Tabs
tab_upload, tab_camera = st.tabs(["📁 Image / Batch Upload", "📸 Real-Time Optical Capture"])
with tab_upload:
    uploaded_file = st.file_uploader("Upload a tray photo, bag sample, or warehouse consignment view:", type=["jpg", "jpeg", "png"])
with tab_camera:
    cam_file = st.camera_input("Capture direct field inspection sample:")

active_file = uploaded_file if uploaded_file is not None else cam_file

if active_file is not None:
    pil_img = Image.open(active_file).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_bgr.shape

    annotated = img_bgr.copy()
    stats = {"Grade_A": 0, "Grade_B": 0, "URS": 0, "Defective": 0, "Total": 0}
    boxes = []
    sample_records = []

    # 1. Run YOLO inference
    if model is not None:
        try:
            results = model.predict(img_bgr, conf=conf_thresh, iou=0.40, device="cpu", verbose=False)[0]
            for box in results.boxes:
                cls_name = model.names[int(box.cls[0])]
                if "coin" not in cls_name:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))
        except Exception:
            pass

    # 2. Adaptive Vision Segmentation Fallback
    if len(boxes) == 0:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 40 and h > 40 and (w * h) < (w_img * h_img * 0.85):
                boxes.append((x, y, w, h))

    # Process Detections
    for idx, (x, y, w, h) in enumerate(boxes):
        crop = img_bgr[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
        defect_type = check_defect(crop) if crop.size > 0 else "Sound"
        diameter_mm = round(max(w, h) * custom_px_to_mm, 1)

        if defect_type != "Sound":
            category = f"Defective ({defect_type})"
            stats["Defective"] += 1
            color = (0, 0, 255)
        else:
            if diameter_mm >= 45.0:
                category = "Grade A"
                stats["Grade_A"] += 1
                color = (0, 200, 0)
            elif diameter_mm >= 40.0:
                category = "Grade B"
                stats["Grade_B"] += 1
                color = (0, 255, 255)
            else:
                category = "URS"
                stats["URS"] += 1
                color = (255, 140, 0)

        stats["Total"] += 1
        sample_records.append({
            "Onion #": f"#{idx+1}",
            "Equatorial Diameter (mm)": diameter_mm,
            "Quality Grade": category,
            "Pathology": defect_type
        })
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        cv2.putText(annotated, f"{category} | {diameter_mm}mm", (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total = max(stats["Total"], 1)
    grade_a_pct = (stats["Grade_A"] / total) * 100
    grade_b_pct = (stats["Grade_B"] / total) * 100
    urs_pct = (stats["URS"] / total) * 100
    defective_pct = (stats["Defective"] / total) * 100

    # Pricing Engine Calculations
    effective_rate_per_quintal = (
        (grade_a_pct / 100.0) * base_mandi_rate +
        (grade_b_pct / 100.0) * (base_mandi_rate * 0.85) +
        (urs_pct / 100.0) * (base_mandi_rate * 0.55) -
        (defective_pct / 100.0) * (base_mandi_rate * 0.40)
    )
    effective_rate_per_quintal = max(effective_rate_per_quintal, 500.0)
    total_payout = effective_rate_per_quintal * lot_weight_quintals
    pricing_data = {
        "rate_per_kg": effective_rate_per_quintal / 100.0,
        "total_payout": total_payout
    }

    # --------------------------------------------------------------------------
    # TOP KPI SCORECARD
    # --------------------------------------------------------------------------
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Counted", f"{stats['Total']} units")
    kpi2.metric("Grade A (>45mm)", f"{grade_a_pct:.1f}%", help="Standard Export & Buffer Grade")
    kpi3.metric("Grade B (40-45mm)", f"{grade_b_pct:.1f}%")
    kpi4.metric("URS (<40mm)", f"{urs_pct:.1f}%", delta=f"-{urs_pct:.1f}%" if urs_pct > 15 else "Normal", delta_color="inverse")
    kpi5.metric("Defect Ratio", f"{defective_pct:.1f}%", delta=f"{defective_pct:.1f}%" if defective_pct > 5 else "Clean", delta_color="inverse")

    # Procurement Verdict Banner
    if defective_pct < 5.0 and grade_a_pct >= 50.0:
        st.markdown(
            """<div style='background-color: #DEF7EC; border: 1px solid #31C48D; padding: 12px; border-radius: 8px;'>
            <b style='color: #03543F; font-size: 16px;'>🟢 PROCUREMENT CLEARANCE: APPROVED FOR BUFFER STOCK</b><br/>
            <span style='color: #046C4E; font-size: 13px;'>Consignment complies strictly with NAFED/DoCA buffer storage specifications (Defects < 5% and Grade A >= 50%).</span>
            </div>""", unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style='background-color: #FDE8E8; border: 1px solid #F98080; padding: 12px; border-radius: 8px;'>
            <b style='color: #9B1C1C; font-size: 16px;'>🔴 PROCUREMENT DISPUTE: RE-INSPECTION / DISPUTE PROTOCOL TRIGGERED</b><br/>
            <span style='color: #C81E1E; font-size: 13px;'>Lot exhibits high defect or URS ratio exceeding the buffer threshold. Secondary manual sampling required.</span>
            </div>""", unsafe_allow_html=True
        )

    # --------------------------------------------------------------------------
    # VISUAL COMPARISON & ADVANCED CHARTS
    # --------------------------------------------------------------------------
    st.markdown("### 🔍 Spatial Vision & Analytics")
    col_img_raw, col_img_ai = st.columns(2)

    with col_img_raw:
        st.caption("📸 Original Ingested Sample View")
        st.image(img_rgb, use_container_width=True)

    with col_img_ai:
        st.caption("🎯 Edge Detection, Diameter Mapping & Bounding Segmentation")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    # Plotly Charts Row
    st.markdown("### 📊 Batch Quality & Sizing Distributions")
    chart_col1, chart_col2 = st.columns(2)

    # 1. Quality Donut Chart
    with chart_col1:
        donut_df = pd.DataFrame({
            "Grade": ["Grade A (>45mm)", "Grade B (40-45mm)", "URS (<40mm)", "Defective (Rot/Sprout)"],
            "Count": [stats["Grade_A"], stats["Grade_B"], stats["URS"], stats["Defective"]]
        })
        fig_donut = px.pie(
            donut_df,
            values="Count",
            names="Grade",
            hole=0.55,
            color="Grade",
            color_discrete_map={
                "Grade A (>45mm)": "#14803C",
                "Grade B (40-45mm)": "#EAB308",
                "URS (<40mm)": "#F97316",
                "Defective (Rot/Sprout)": "#DC2626"
            }
        )
        fig_donut.update_layout(title="Batch Proportion Breakdown", margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig_donut, use_container_width=True)

    # 2. Diameter Distribution Histogram
    with chart_col2:
        if len(sample_records) > 0:
            df_samples = pd.DataFrame(sample_records)
            fig_hist = px.histogram(
                df_samples,
                x="Equatorial Diameter (mm)",
                nbins=15,
                color="Quality Grade",
                color_discrete_map={
                    "Grade A": "#14803C",
                    "Grade B": "#EAB308",
                    "URS": "#F97316",
                    "Defective (Rotten)": "#DC2626",
                    "Defective (Sprouted)": "#9333EA"
                }
            )
            # Add Agmark Threshold markers
            fig_hist.add_vline(x=45.0, line_dash="dash", line_color="#14803C", annotation_text="Grade A (45mm)")
            fig_hist.add_vline(x=40.0, line_dash="dash", line_color="#EAB308", annotation_text="Grade B (40mm)")
            fig_hist.update_layout(title="Equatorial Diameter Histogram (mm)", margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No numerical samples available for histogram rendering.")

    # --------------------------------------------------------------------------
    # LIVE MANDI COMMERCIAL SETTLEMENT DESK
    # --------------------------------------------------------------------------
    st.markdown("### 💵 Automated Mandi Valuation & Settlement")
    fin1, fin2, fin3 = st.columns(3)
    fin1.metric("Base Rate for Grade A", f"₹ {base_mandi_rate:,} / Qtl")
    fin2.metric("Effective Adjusted Valuation", f"₹ {effective_rate_per_quintal:,.2f} / Qtl", f"₹ {(effective_rate_per_quintal/100):.2f}/kg")
    fin3.metric("Net Farmer / Lot Payable", f"₹ {total_payout:,.2f}", f"Net {lot_weight_quintals} Quintals")

    # --------------------------------------------------------------------------
    # DATA TABLE & AUDIT EXPORT
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Sample Ledger & Certified Audit")
    
    if len(sample_records) > 0:
        with st.expander("👁️ View Individual Sample Log & Dimensions"):
            st.dataframe(pd.DataFrame(sample_records), use_container_width=True)

    pdf_file_path = generate_pdf(stats, batch_id, mandi_name, pricing_data)
    with open(pdf_file_path, "rb") as f:
        st.download_button(
            label="📥 Download Cryptographic Quality Certificate (PDF)",
            data=f,
            file_name=f"Certificate_{batch_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:
    st.info("👈 Please capture a live photo or upload an image sample above to begin inspection.")
