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

st.set_page_config(page_title="KandaDrishti AI", page_icon="🧅", layout="centered")

st.title("🧅 KandaDrishti AI")
st.subheader("Automated Onion Quality Grading & Sizing")
st.caption("Department of Consumer Affairs (DoCA) | Smart India Hackathon")

@st.cache_resource
def load_yolo_model():
    if os.path.exists("best.pt"):
        try:
            return YOLO("best.pt")
        except Exception:
            return None
    return None

model = load_yolo_model()

def generate_pdf(stats, batch_id="DOCA-2026-SIH"):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "KandaDrishti AI - Quality Assessment Certificate")
    c.setFont("Helvetica", 10)
    c.drawString(50, 735, f"Authority: Department of Consumer Affairs (DoCA) | Batch: {batch_id}")
    c.drawString(50, 720, f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, 710, 550, 710)

    total = max(stats["Total"], 1)
    grade_a_pct = (stats["Grade_A"] / total) * 100
    grade_b_pct = (stats["Grade_B"] / total) * 100
    urs_pct = (stats["URS"] / total) * 100
    defective_pct = (stats["Defective"] / total) * 100

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 680, "Grading Breakdown (Agmark / NAFED Standards):")
    c.setFont("Helvetica", 11)
    c.drawString(70, 650, f"Total Count Inspected: {stats['Total']} units")
    c.drawString(70, 625, f"Grade A (>45 mm equatorial dia): {stats['Grade_A']} ({grade_a_pct:.1f}%)")
    c.drawString(70, 600, f"Grade B (40 - 45 mm equatorial dia): {stats['Grade_B']} ({grade_b_pct:.1f}%)")
    c.drawString(70, 575, f"URS (<40 mm Under Regular Size): {stats['URS']} ({urs_pct:.1f}%)")
    c.drawString(70, 550, f"Defective (Rotten / Sprouted): {stats['Defective']} ({defective_pct:.1f}%)")
    c.line(50, 520, 550, 520)

    c.setFont("Helvetica-Bold", 12)
    status = "ACCEPTED FOR BUFFER STOCK PROCUREMENT" if (defective_pct < 5.0 and grade_a_pct >= 50.0) else "DISPUTED / MANUAL RE-INSPECTION REQUIRED"
    c.drawString(50, 490, f"Procurement Verdict: {status}")
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

# Input choices for mobile
input_method = st.radio("Choose Input Mode:", ["📁 Upload File/Photo", "📸 Use Phone Camera"], horizontal=True)

if input_method == "📁 Upload File/Photo":
    uploaded_file = st.file_uploader("Select an onion image:", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Capture an onion sample:")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_bgr.shape

    annotated = img_bgr.copy()
    stats = {"Grade_A": 0, "Grade_B": 0, "URS": 0, "Defective": 0, "Total": 0}
    boxes = []

    if model is not None:
        try:
            results = model.predict(img_bgr, conf=0.04, iou=0.40, device="cpu", verbose=False)[0]
            for box in results.boxes:
                cls_name = model.names[int(box.cls[0])]
                if "coin" not in cls_name:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))
        except Exception:
            pass

    if len(boxes) == 0:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 45 and h > 45 and (w * h) < (w_img * h_img * 0.85):
                boxes.append((x, y, w, h))

    px_to_mm = 0.16

    for (x, y, w, h) in boxes:
        crop = img_bgr[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
        defect_type = check_defect(crop) if crop.size > 0 else "Sound"
        diameter_mm = round(max(w, h) * px_to_mm, 1)

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
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        cv2.putText(annotated, f"{category} | {diameter_mm}mm", (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total = max(stats["Total"], 1)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Graded & Sized Output", use_container_width=True)

    st.markdown("### 📊 Batch Metrics Summary")
    col1, col2 = st.columns(2)
    col1.metric("Total Count", stats['Total'])
    col1.metric("Grade A (>45mm)", f"{(stats['Grade_A']/total)*100:.1f}%")
    col2.metric("Grade B (40-45mm)", f"{(stats['Grade_B']/total)*100:.1f}%")
    col2.metric("Defective Ratio", f"{(stats['Defective']/total)*100:.1f}%")

    pdf_path = generate_pdf(stats)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Official Quality Certificate (PDF)",
            data=f,
            file_name="Onion_Quality_Certificate.pdf",
            mime="application/pdf",
            use_container_width=True
        )
