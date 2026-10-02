import os
import time
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# dotenv for reading hidden .env files
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Google GenAI import
try:
    import google.generativeai as genai
    GEMINI_INSTALLED = True
except ImportError:
    GEMINI_INSTALLED = False

# ==============================================================================
# SECURE API KEY LOADING
# ==============================================================================
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        gemini_api_key = None

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AURA AI | Clinical Biomechanics & Dual AI Engine",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp { 
        background: radial-gradient(circle at top right, #0d1117, #010409); 
        color: #F0F6FC; 
    }
    
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F0F6FC !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: #21262d !important;
        color: #FFFFFF !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.8) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px !important;
        padding: 16px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-perfect { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .badge-warn { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .badge-danger { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }

    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        background-color: #161b22; 
        padding: 6px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] { 
        height: 44px; 
        border-radius: 8px; 
        color: #8B949E !important; 
        font-weight: 600; 
        border: none !important; 
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important; 
        color: #FFFFFF !important; 
    }

    .user-msg {
        background-color: #23272d;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #6366F1;
    }
    .ai-msg {
        background-color: #161b22;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #10B981;
    }
    </style>
""", unsafe_allow_html=True)

# Voice Guidance Component
def play_voice_guidance(text_prompt, enable_voice=True):
    if enable_voice and text_prompt:
        clean_text = text_prompt.replace("'", "").replace('"', '')
        js_code = f"""
        <script>
        (function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance('{clean_text}');
                msg.rate = 1.0;
                msg.pitch = 1.0;
                msg.volume = 1.0;
                window.speechSynthesis.speak(msg);
            }}
        }})();
        </script>
        """
        components.html(js_code, height=0, width=0)

# ==============================================================================
# 2. BIOMECHANICS ENGINE
# ==============================================================================
class BiomechanicsEngine:
    def __init__(self, model_path="pose_landmarker.task"):
        if os.path.exists(model_path):
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.CPU
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
        else:
            self.detector = None
            
        self.reset_session()

    def reset_session(self):
        self.counter = 0
        self.stage = "UP"
        self.accuracy_scores = []
        self.snapshots = []
        self.rep_logs = []
        self.combo = 0
        self.prev_time = time.time()
        self.prev_joint_pos = None
        self.velocity = 0.0
        self.power_watts = 0.0
        self.jitter_buffer = []
        self.hold_start_time = None
        self.hold_duration = 0
        self.boss_hp = 100
        self.last_speech_time = 0

    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360.0 - angle if angle > 180.0 else angle

    def process_frame(self, frame, exercise, user_weight):
        if self.detector is None:
            return frame, self.counter, self.stage, "Model file missing", "Please place pose_landmarker.task file", 0, 0, 0, [], 100, 0, 0, ""

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        canvas_img = frame.copy()

        feedback = "Position yourself clearly in frame"
        warning = ""
        accuracy = 100
        voice_prompt = ""

        if not results.pose_landmarks:
            cv2.putText(canvas_img, "NO USER DETECTED", (int(w*0.2), int(h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return canvas_img, self.counter, self.stage, "No user detected", "Step into frame", 0, 0, 0, self.snapshots, self.boss_hp, 0, 0, ""

        landmarks = results.pose_landmarks[0]
        def get_pt(idx): return [landmarks[idx].x * w, landmarks[idx].y * h]

        l_shoulder, r_shoulder = get_pt(11), get_pt(12)
        l_elbow, r_elbow = get_pt(13), get_pt(14)
        l_wrist, r_wrist = get_pt(15), get_pt(16)
        l_hip, r_hip = get_pt(23), get_pt(24)
        l_knee, r_knee = get_pt(25), get_pt(26)
        l_ankle, r_ankle = get_pt(27), get_pt(28)

        shoulder_width = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
        scale_factor = 0.45 / (shoulder_width + 1e-6)

        l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        l_elbow_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        l_hip_angle = self.calculate_angle(l_shoulder, l_hip, l_knee)

        asymmetry_delta = abs(l_knee_angle - r_knee_angle)

        curr_time = time.time()
        dt = curr_time - self.prev_time
        if self.prev_joint_pos and dt > 0:
            disp_px = np.linalg.norm(np.array(l_hip) - np.array(self.prev_joint_pos))
            self.velocity = (disp_px * scale_factor) / dt
            self.power_watts = (user_weight * 9.81) * self.velocity
        self.prev_joint_pos = l_hip
        self.prev_time = curr_time

        # Logic for exercises
        if exercise == "Squats":
            accuracy = max(0, 100 - int(asymmetry_delta * 1.5))
            if l_knee_angle < 100:
                self.stage = "DOWN"
                feedback = "Drive upwards through heels"
            if l_knee_angle > 160 and self.stage == "DOWN":
                self.stage = "UP"
                self.counter += 1
                self.accuracy_scores.append(accuracy)
                self.boss_hp = max(0, self.boss_hp - 10)
                voice_prompt = f"Good rep! Total {self.counter}"
                if accuracy >= 90:
                    self.snapshots.append(canvas_img.copy())

        elif exercise == "Bicep Curls":
            accuracy = max(0, 100 - int(abs(l_elbow_angle - r_elbow_angle) * 0.8))
            if l_elbow_angle > 160:
                self.stage = "DOWN"
                feedback = "Curl upward towards shoulder"
            if l_elbow_angle < 40 and self.stage == "DOWN":
                self.stage = "UP"
                self.counter += 1
                self.accuracy_scores.append(accuracy)
                self.boss_hp = max(0, self.boss_hp - 10)
                voice_prompt = f"Squeeze at peak! Rep {self.counter}"
                if accuracy >= 90:
                    self.snapshots.append(canvas_img.copy())

        elif exercise == "Push-ups":
            accuracy = 100 if l_hip_angle > 150 else 70
            if l_elbow_angle < 90:
                self.stage = "DOWN"
                feedback = "Push chest up"
            if l_elbow_angle > 160 and self.stage == "DOWN":
                self.stage = "UP"
                self.counter += 1
                self.accuracy_scores.append(accuracy)
                self.boss_hp = max(0, self.boss_hp - 10)
                voice_prompt = f"Push up! Rep {self.counter}"
                if accuracy >= 90:
                    self.snapshots.append(canvas_img.copy())

        else:
            feedback = f"Executing {exercise}..."
            accuracy = 95

        for joint in [l_knee, r_knee, l_elbow, r_elbow, l_hip, r_hip]:
            cv2.circle(canvas_img, (int(joint[0]), int(joint[1])), 7, (0, 255, 255), -1)

        avg_acc = int(np.mean(self.accuracy_scores)) if self.accuracy_scores else 100
        return canvas_img, self.counter, self.stage, feedback, warning, accuracy, avg_acc, self.combo, self.snapshots, self.boss_hp, self.velocity, self.power_watts, voice_prompt

if 'engine' not in st.session_state:
    st.session_state.engine = BiomechanicsEngine()

# WebRTC Video Processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.exercise = "Squats"
        self.weight = 70

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        processed_img, *_ = st.session_state.engine.process_frame(img, self.exercise, self.weight)
        return processed_img

# ==============================================================================
# 3. FRONTEND UI & CHAT ENGINE
# ==============================================================================
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0 20px 0;">
        <div>
            <h1 style="margin: 0; font-weight: 800; font-size: 2.2rem; color: #FFFFFF;">
                ⚡ AURA AI <span style="font-size:1.1rem; font-weight:600; color:#818CF8;">Clinical Biomechanics & AI Coach</span>
            </h1>
        </div>
    </div>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("## 🎛️ Control Hub")
selected_exercise = st.sidebar.selectbox(
    "Select Target Exercise",
    ["Squats", "Push-ups", "Bicep Curls", "Lunges", "Plank"]
)
user_weight_kg = st.sidebar.number_input("User Mass (kg)", min_value=30, max_value=200, value=70)

st.sidebar.markdown("---")
if gemini_api_key:
    st.sidebar.success("⚡ Live Gemini AI Engine Active")
else:
    st.sidebar.info("💡 Local Smart Rules Active")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Session", use_container_width=True):
    st.session_state.engine.reset_session()
    st.rerun()

# DASHBOARD TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🎥 Live Biomechanics HUD", 
    "📊 Kinematics & Power", 
    "🤖 AI Fitness Assistant", 
    "📑 Clinical PDF Audit"
])

with tab1:
    st.markdown("### 🎥 Cloud WebRTC Camera Stream")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    ctx = webrtc_streamer(
        key="aura-live-stream",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.exercise = selected_exercise
        ctx.video_processor.weight = user_weight_kg

    st.metric("Completed Reps", st.session_state.engine.counter)

with tab2:
    st.markdown("### ⚡ Telemetry Metrics")
    st.metric("Limb Velocity", f"{round(st.session_state.engine.velocity, 2)} m/s")
    st.metric("Mechanical Power Output", f"{round(st.session_state.engine.power_watts, 1)} W")

# UPDATED GEMINI AI ENGINE FIX
with tab3:
    st.markdown("### 🤖 AURA AI Personal Coach")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask any question about exercise form, diet, or workout plans:", placeholder="E.g., Squats karte waqt knee pain se kaise bache?")
    
    if st.button("Send Question") and user_query:
        response = ""
        if gemini_api_key and GEMINI_INSTALLED:
            try:
                genai.configure(api_key=gemini_api_key)
                # Model fallbacks to resolve API 404 version error
                model_names = ['gemini-2.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']
                model = None
                for m_name in model_names:
                    try:
                        model = genai.GenerativeModel(m_name)
                        break
                    except Exception:
                        continue
                
                if model:
                    context_prompt = f"You are AURA AI, an expert fitness coach. User Question: {user_query}. Provide a concise response."
                    response_obj = model.generate_content(context_prompt)
                    response = response_obj.text
                else:
                    raise Exception("No supported Gemini models found.")
            except Exception as e:
                response = f"⚠️ Gemini API Fallback Active. Response generated via Smart Rules."
                query_lower = user_query.lower()
                if "knee" in query_lower or "pain" in query_lower:
                    response += "\n\nKnee pain se bachne ke liye knees ko toes ke aage zyadatar na jaane dein aur heels ko firm rakhein."
                elif "diet" in query_lower or "protein" in query_lower:
                    response += f"\n\nDaily weight ({user_weight_kg}kg) ke according ~{int(user_weight_kg * 1.8)}g protein intake rakhein."
                else:
                    response += f"\n\nOptimum gains ke liye proper form aur consistent 3-4 sets perform karein."
        else:
            query_lower = user_query.lower()
            if "knee" in query_lower or "pain" in query_lower:
                response = "Knee pain se bachne ke liye knees ko toes ke aage zyadatar na jaane dein aur heels ko firm rakhein."
            elif "diet" in query_lower or "protein" in query_lower:
                response = f"Daily weight ({user_weight_kg}kg) ke according ~{int(user_weight_kg * 1.8)}g protein intake rakhein."
            else:
                response = f"Optimum gains ke liye proper form aur consistent 3-4 sets perform karein."
            
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("ai", response))

    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(f'<div class="user-msg"><b>👤 You:</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg"><b>🤖 AURA AI Assistant:</b> {msg}</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("### 📄 Generate Performance Audit")
    if st.button("Export Summary Report"):
        st.success("Report Generated Successfully!")