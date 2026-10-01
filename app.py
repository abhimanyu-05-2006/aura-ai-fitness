import os
import io
import math
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# dotenv for reading hidden .env files locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Google GenAI import (Error-handled)
try:
    import google.generativeai as genai
    GEMINI_INSTALLED = True
except ImportError:
    GEMINI_INSTALLED = False

# ==============================================================================
# SECURE API KEY LOADING (ENV / SECRETS MANAGER)
# ==============================================================================
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        gemini_api_key = None

if gemini_api_key and GEMINI_INSTALLED:
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.warning(f"Gemini configuration error: {e}")

# ==============================================================================
# 1. PAGE & ENGINE CONFIGURATION (PRO DARK UI / HIGH CONTRAST)
# ==============================================================================
st.set_page_config(
    page_title="AURA AI | Clinical Biomechanics & Dual AI Engine",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Global App Dark Theme Background */
    .stApp { 
        background: radial-gradient(circle at top right, #0d1117, #010409); 
        color: #F0F6FC; 
    }
    
    /* SIDEBAR HIGH-CONTRAST FIX */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span {
        color: #F0F6FC !important;
        font-weight: 600 !important;
    }

    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #58A6FF !important;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BIOMECHANICS ENGINE (MEDIAPIPE CPU DELEGATE OPTIMIZED)
# ==============================================================================
class BiomechanicsEngine:
    def __init__(self, model_path="pose_landmarker.task"):
        if not os.path.exists(model_path):
            st.error(f"❌ Model file '{model_path}' not found in working directory. Please ensure it is tracked in Git.")
            st.stop()
            
        # Explicit CPU Delegate to avoid ctypes/EGL crashes on headless Linux
        base_options = BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU
        )
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates 2D angle between three points (a-b-c) in degrees."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return round(float(angle), 2)

    def evaluate_pose(self, image_np):
        """Processes RGB Image array and returns biomechanical metrics."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.pose_landmarks:
            return None, "No human pose detected. Please position yourself clearly in frame."
            
        landmarks = detection_result.pose_landmarks[0]
        
        # Keypoints Extraction (Squats & General Biomechanics)
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]
        
        right_hip = landmarks[24]
        right_knee = landmarks[26]
        right_ankle = landmarks[28]
        
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]

        # Knee Flexion Angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_flexion = round((left_knee_angle + right_knee_angle) / 2.0, 2)
        
        # Elbow Flexion
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Assessment Logic
        score = 100
        feedback = []
        
        if avg_knee_flexion > 120:
            score -= 25
            feedback.append("Depth insufficient: Lower hips deeper for full squat range.")
        elif avg_knee_flexion < 70:
            score -= 10
            feedback.append("Excessive squat depth: Ensure joint stability at lowest point.")
        else:
            feedback.append("Optimal squat depth achieved.")
            
        return {
            "landmarks": landmarks,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "avg_knee_flexion": avg_knee_flexion,
            "left_elbow_angle": left_elbow_angle,
            "score": max(0, score),
            "feedback": feedback
        }, None

# Initialize Engine once in Session State
if "engine" not in st.session_state:
    st.session_state.engine = BiomechanicsEngine()

# ==============================================================================
# 3. PDF REPORT GENERATOR (REPORTLAB INTEGRATION)
# ==============================================================================
def generate_pdf_report(metrics_data, coaching_notes=""):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, 750, "AURA AI — Biomechanical Audit Report")
    
    p.setFont("Helvetica", 10)
    p.drawString(50, 735, "Clinical Movement Analysis & Form Evaluation")
    p.line(50, 725, 560, 725)
    
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, 690, f"Overall Pose Score: {metrics_data.get('score', 0)} / 100")
    
    p.setFont("Helvetica", 11)
    p.drawString(50, 660, f"• Left Knee Angle: {metrics_data.get('left_knee_angle', 'N/A')}°")
    p.drawString(50, 640, f"• Right Knee Angle: {metrics_data.get('right_knee_angle', 'N/A')}°")
    p.drawString(50, 620, f"• Average Knee Flexion: {metrics_data.get('avg_knee_flexion', 'N/A')}°")
    p.drawString(50, 600, f"• Left Elbow Angle: {metrics_data.get('left_elbow_angle', 'N/A')}°")
    
    p.drawString(50, 560, "Biomechanical Feedback:")
    y = 540
    for note in metrics_data.get('feedback', []):
        p.drawString(70, y, f"- {note}")
        y -= 20
        
    if coaching_notes:
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y - 20, "AI Coach Insights:")
        p.setFont("Helvetica", 10)
        p.drawString(50, y - 40, coaching_notes[:300])
        
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ==============================================================================
# 4. USER INTERFACE & DUAL AI DASHBOARD
# ==============================================================================
st.title("⚡ AURA AI: Biomechanics & Yoga Evaluator")
st.caption("Real-time Local MediaPipe Computer Vision Engine + Hybrid Gemini API Coaching")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Engine Control Pane")
    exercise_type = st.selectbox(
        "Select Exercise Module",
        ["Barbell Squats", "Push-ups", "Virabhadrasana II (Warrior II)", "Trikonasana (Triangle Pose)"]
    )
    
    st.markdown("---")
    st.subheader("🔑 API & Environment Status")
    if gemini_api_key:
        st.success("Gemini API Key Loaded Securely")
    else:
        st.error("Gemini API Key Missing! Set in `.env` or Streamlit Secrets.")

# Main Application Tabs
tab1, tab2 = st.tabs(["📊 Motion Evaluation", "🤖 AI Fitness Assistant"])

with tab1:
    st.subheader("Pose Capture & Analysis")
    uploaded_file = st.file_uploader("Upload an Image / Keyframe for Evaluation", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        import cv2
        image_np = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image_rgb, caption="Source Input", use_container_width=True)
            
        with col2:
            with st.spinner("Processing Biomechanics..."):
                results, error = st.session_state.engine.evaluate_pose(image_rgb)
                
                if error:
                    st.error(error)
                else:
                    st.success("Pose Evaluated Successfully!")
                    st.metric("Form Score", f"{results['score']} / 100")
                    
                    st.write("**Biomechanical Metrics:**")
                    st.write(f"- **Avg Knee Flexion:** {results['avg_knee_flexion']}°")
                    st.write(f"- **Left Knee Angle:** {results['left_knee_angle']}°")
                    st.write(f"- **Right Knee Angle:** {results['right_knee_angle']}°")
                    
                    st.write("**Feedback:**")
                    for fb in results['feedback']:
                        st.info(fb)
                        
                    # Generate PDF Download Button
                    pdf_buffer = generate_pdf_report(results)
                    st.download_button(
                        label="📄 Download Biomechanics PDF Report",
                        data=pdf_buffer,
                        file_name="Biomechanics_Audit_Report.pdf",
                        mime="application/pdf"
                    )

with tab2:
    st.subheader("🤖 Dual AI Fitness & Recovery Coach")
    if not gemini_api_key or not GEMINI_INSTALLED:
        st.warning("Gemini API is not configured. Please add `GEMINI_API_KEY` to Streamlit Secrets.")
    else:
        user_query = st.text_input("Ask about form correction, workout plans, or injury prevention:")
        if st.button("Ask AI Coach"):
            if user_query:
                with st.spinner("Gemini is analyzing query..."):
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        prompt = f"You are AURA AI, an elite clinical biomechanics expert and yoga coach. Answer concisely: {user_query}"
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error querying Gemini API: {e}")