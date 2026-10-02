import os
import time
import cv2
import tempfile
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, VideoHTMLAttributes

# Real-time UI sync refresh component
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ------------------------------------------------------------------------------
# STREAMLIT CLOUD PERMISSION & CRASH-PROOF MEDIAPIPE LOADER
# ------------------------------------------------------------------------------
os.environ["MEDIAPIPE_CACHE_DIR"] = tempfile.gettempdir()

import mediapipe as mp

# Safe Dynamic MediaPipe Loader
mp_solutions = getattr(mp, "solutions", None)

if mp_solutions is not None:
    import mediapipe as mp

# Absolute Fail-Safe MediaPipe Solutions Loader
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Backup resolution for specific virtualenv configurations
    from mediapipe import python as mp_python  # type: ignore
    mp_pose = mp_python.solutions.pose
    mp_drawing = mp_python.solutions.drawing_utils

GEMINI_INSTALLED = False
try:
    import google.generativeai as genai
    GEMINI_INSTALLED = True
except ImportError:
    try:
        from google import genai
        GEMINI_INSTALLED = True
    except ImportError:
        GEMINI_INSTALLED = False

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        gemini_api_key = None

# ==============================================================================
# 1. PAGE & ENGINE CONFIGURATION
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
        margin-top: 10px;
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
# 2. LIGHTWEIGHT BIOMECHANICS POSE ENGINE
# ==============================================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

class UltraFastPoseProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        self.exercise = "Squats"
        self.weight = 70
        self.counter = 0
        self.stage = "UP"
        self.last_acc = 100
        self.last_feedback = "Position yourself clearly in frame"
        self.last_warning = ""
        self.velocity = 0.0
        self.power_watts = 0.0
        self.boss_hp = 100
        self.hold_duration = 0
        self.hold_start_time = None
        self.prev_time = time.time()
        self.prev_hip = None
        self.accuracy_scores = []
        self.snapshots = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        self.last_warning = ""

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            lm = results.pose_landmarks.landmark
            def get_coord(idx): return [lm[idx].x * w, lm[idx].y * h]

            l_shoulder, r_shoulder = get_coord(11), get_coord(12)
            l_elbow, r_elbow = get_coord(13), get_coord(14)
            l_wrist, r_wrist = get_coord(15), get_coord(16)
            l_hip, r_hip = get_coord(23), get_coord(24)
            l_knee, r_knee = get_coord(25), get_coord(26)
            l_ankle, r_ankle = get_coord(27), get_coord(28)

            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)

            asym_delta = abs(l_knee_angle - r_knee_angle)

            curr_time = time.time()
            dt = curr_time - self.prev_time
            if self.prev_hip and dt > 0:
                disp_px = np.linalg.norm(np.array(l_hip) - np.array(self.prev_hip))
                self.velocity = (disp_px * 0.002) / dt
                self.power_watts = (self.weight * 9.81) * self.velocity
            self.prev_hip = l_hip
            self.prev_time = curr_time

            if self.exercise == "Squats":
                self.last_acc = max(30, 100 - int(asym_delta * 1.5))
                if l_knee[0] > l_ankle[0] + 25:
                    self.last_warning = "Knee Shear Stress High!"
                    self.last_acc -= 20

                if l_knee_angle < 100:
                    self.stage = "DOWN"
                    self.last_feedback = "Drive upwards through heels"
                if l_knee_angle > 160 and self.stage == "DOWN":
                    self.stage = "UP"
                    self.counter += 1
                    self.accuracy_scores.append(self.last_acc)
                    self.boss_hp = max(0, self.boss_hp - 10)
                    self.last_feedback = f"Good Squat! Rep #{self.counter}"
                    if self.last_acc >= 85 and len(self.snapshots) < 6:
                        self.snapshots.append(img.copy())

            elif self.exercise == "Bicep Curls":
                self.last_acc = max(30, 100 - int(abs(l_elbow_angle - r_elbow_angle) * 0.8))
                if l_elbow_angle > 160:
                    self.stage = "DOWN"
                    self.last_feedback = "Curl upward towards shoulder"
                if l_elbow_angle < 40 and self.stage == "DOWN":
                    self.stage = "UP"
                    self.counter += 1
                    self.accuracy_scores.append(self.last_acc)
                    self.boss_hp = max(0, self.boss_hp - 10)
                    self.last_feedback = f"Squeeze! Rep #{self.counter}"
                    if self.last_acc >= 85 and len(self.snapshots) < 6:
                        self.snapshots.append(img.copy())

            elif self.exercise == "Push-ups":
                self.last_acc = 100 if l_hip_angle > 150 else 70
                if l_hip_angle <= 150:
                    self.last_warning = "Hips Sagging!"
                
                if l_elbow_angle < 90:
                    self.stage = "DOWN"
                    self.last_feedback = "Push chest up"
                if l_elbow_angle > 160 and self.stage == "DOWN":
                    self.stage = "UP"
                    self.counter += 1
                    self.accuracy_scores.append(self.last_acc)
                    self.boss_hp = max(0, self.boss_hp - 10)
                    self.last_feedback = f"Push Up! Rep #{self.counter}"
                    if self.last_acc >= 85 and len(self.snapshots) < 6:
                        self.snapshots.append(img.copy())

            elif self.exercise in ["Plank", "Warrior II (Yoga)", "Tree Pose (Yoga)"]:
                if not self.hold_start_time: 
                    self.hold_start_time = time.time()
                self.hold_duration = int(time.time() - self.hold_start_time)
                self.last_acc = 95
                self.last_feedback = f"Holding Position... Time: {self.hold_duration}s"

            elif self.exercise == "Lunges":
                self.last_acc = max(30, 100 - int(asym_delta * 1.2))
                if l_knee_angle < 90 or r_knee_angle < 90:
                    self.stage = "DOWN"
                    self.last_feedback = "Step back up to standing"
                if l_knee_angle > 160 and r_knee_angle > 160 and self.stage == "DOWN":
                    self.stage = "UP"
                    self.counter += 1
                    self.accuracy_scores.append(self.last_acc)
                    self.boss_hp = max(0, self.boss_hp - 10)
                    self.last_feedback = f"Lunge complete! Rep #{self.counter}"
                    if self.last_acc >= 85 and len(self.snapshots) < 6:
                        self.snapshots.append(img.copy())

        else:
            self.last_feedback = "No person detected in frame"

        cv2.putText(img, f"REPS: {self.counter}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"FORM: {self.last_acc}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

# ==============================================================================
# 3. FRONTEND UI & DASHBOARD
# ==============================================================================
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0 20px 0;">
        <div>
            <h1 style="margin: 0; font-weight: 800; font-size: 2.2rem; color: #FFFFFF;">
                ⚡ AURA AI <span style="font-size:1.1rem; font-weight:600; color:#818CF8;">Clinical Biomechanics & AI Coach</span>
            </h1>
            <p style="margin: 4px 0 0 0; color: #9CA3AF; font-size: 0.9rem;">
                Real-time joint shear stress, live voice coaching, and Hybrid AI fitness guidance.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🎛 Control Hub")

selected_exercise = st.sidebar.selectbox(
    "Select Target Exercise",
    ["Squats", "Push-ups", "Bicep Curls", "Lunges", "Plank", "Warrior II (Yoga)", "Tree Pose (Yoga)"]
)

user_weight_kg = st.sidebar.number_input("User Mass (kg)", min_value=30, max_value=200, value=70)

st.sidebar.markdown("---")
if gemini_api_key:
    st.sidebar.success("⚡ Live Gemini AI Engine Active")
else:
    st.sidebar.info("💡 Local Smart Rules Active (No Key Needed)")

st.sidebar.markdown("---")
enable_voice_coach = st.sidebar.toggle("Enable Voice Guidance", value=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 Live Biomechanics HUD", 
    "📊 Kinematics & Power", 
    "🤖 AI Fitness Assistant", 
    "📸 Flawless Rep Snapshots", 
    "📑 Clinical PDF Audit"
])

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]
})

with tab1:
    col_cam, col_hud = st.columns([2.2, 1])
    
    with col_cam:
        ctx = webrtc_streamer(
            key="aura-webrtc-stream-v2",
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=UltraFastPoseProcessor,
            media_stream_constraints={"video": True, "audio": False},
            video_html_attributes=VideoHTMLAttributes(
                autoPlay=True, controls=False, style={"width": "100%"}, muted=True
            ),
            async_transform=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.exercise = selected_exercise
            ctx.video_processor.weight = user_weight_kg

    with col_hud:
        st.markdown("### 📈 Live Telemetry")
        
        if ctx.state.playing and HAS_AUTOREFRESH:
            st_autorefresh(interval=500, key="hud_sync_refresh")

        if ctx.video_processor:
            proc = ctx.video_processor
            is_hold = selected_exercise in ["Plank", "Tree Pose (Yoga)", "Warrior II (Yoga)"]
            
            st.metric("Completed Reps / Hold Time", f"{proc.hold_duration}s" if is_hold else proc.counter)
            
            avg_acc = int(np.mean(proc.accuracy_scores)) if proc.accuracy_scores else proc.last_acc
            st.metric("Instant Form Accuracy", f"{proc.last_acc}%", delta=f"Avg: {avg_acc}%")
            
            st.markdown("##### 👾 AI Boss Health")
            st.progress(proc.boss_hp / 100)
            
            if proc.last_acc >= 85:
                st.markdown('<div class="status-badge badge-perfect">🟢 PERFECT BIOMECHANICAL FORM</div>', unsafe_allow_html=True)
            elif proc.last_acc >= 60:
                st.markdown('<div class="status-badge badge-warn">🟡 MINOR POSTURE DEVIATION</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge badge-danger">🔴 HIGH RISK INJURY CORRECTION</div>', unsafe_allow_html=True)

            st.info(f"💡 **AI Guidance**: {proc.last_feedback}")
            if proc.last_warning:
                st.error(f"⚠️ **Hazard Alert**: {proc.last_warning}")
                
            play_voice_guidance(proc.last_feedback, enable_voice=enable_voice_coach)
        else:
            st.warning("⚠️ Press **START** button on camera player to activate webcam stream.")

with tab2:
    st.markdown("### ⚡ Velocity-Based Training (VBT) Metrics")
    if ctx.video_processor:
        proc = ctx.video_processor
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("Limb Velocity", f"{round(proc.velocity, 2)} m/s")
        col_v2.metric("Mechanical Power Output", f"{round(proc.power_watts, 1)} W")
        mech_work_kcal = round((proc.power_watts * 0.1) * 0.000239006, 3)
        col_v3.metric("Mechanical Work Burned", f"{mech_work_kcal} kcal")
    else:
        st.info("Start camera session to view live Kinematics metrics.")

with tab3:
    st.markdown("### 🤖 AURA AI Personal Coach & Assistant")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    age = col_a1.number_input("Age (Years)", min_value=10, max_value=100, value=22)
    height_cm = col_a2.number_input("Height (cm)", min_value=100, max_value=230, value=175)
    fitness_goal = col_a3.selectbox("Primary Fitness Goal", ["Weight Loss", "Muscle Gain", "Endurance & Flexibility", "Posture Correction"])
    activity_level = col_a4.selectbox("Activity Level", ["Beginner", "Intermediate", "Advanced"])

    height_m = height_cm / 100
    bmi = round(user_weight_kg / (height_m ** 2), 1)

    st.markdown("---")
    st.info(f"**BMI:** {bmi} | **Target Goal:** {fitness_goal} | **Level:** {activity_level}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask any question about exercise form, diet, or workout plans:", placeholder="E.g., Squats karte waqt knee pain se kaise bache?")
    
    if st.button("Send Question") and user_query:
        response = ""
        if gemini_api_key and GEMINI_INSTALLED:
            try:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_prompt = f"""
                You are AURA AI, an elite fitness & biomechanics coach.
                User Context: Age {age}, Weight {user_weight_kg}kg, Height {height_cm}cm (BMI: {bmi}), Goal: {fitness_goal}, Exercise: {selected_exercise}.
                Question: {user_query}
                Provide a clear, helpful response in Hinglish or English.
                """
                response_obj = model.generate_content(context_prompt)
                response = response_obj.text
            except Exception as e:
                response = "Knee pain avoid karne ke liye: 1) Knees ko toes ke aage mat jaane do. 2) Heels ground par fix rakho. 3) Warm-up zaroor karein."
        else:
            query_l = user_query.lower()
            if "knee" in query_l or "pain" in query_l:
                response = "Knee pain avoid karne ke liye: 1) Knees ko toes ke aage zyadatar mat jaane do. 2) Squat karte waqt heels ground par rakho. 3) Proper warm up karo."
            else:
                response = "Optimum results ke liye regular form precision aur controlled reps ke sath workout karein."
            
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("ai", response))

    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(f'<div class="user-msg"><b>👤 You:</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg"><b>🤖 AURA AI Assistant:</b> {msg}</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("### 📸 Captured Flawless Form Snapshots (>85% Accuracy)")
    if ctx.video_processor and ctx.video_processor.snapshots:
        snaps = ctx.video_processor.snapshots
        cols = st.columns(3)
        for idx, snap in enumerate(snaps):
            cols[idx % 3].image(snap, channels="BGR", caption=f"Flawless Rep #{idx+1}")
    else:
        st.info("No snapshots captured yet. Complete reps with >85% accuracy to capture snapshots here!")

with tab5:
    st.markdown("### 📄 Generate Clinical Performance Report")
    
    def generate_pdf_report():
        pdf_path = "Biomechanics_Audit_Report.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, 750, "AURA AI - Biomechanics & Posture Audit")
        c.setFont("Helvetica", 11)
        c.drawString(40, 725, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(40, 700, f"Target Exercise: {selected_exercise}")
        if ctx.video_processor:
            proc = ctx.video_processor
            c.drawString(40, 680, f"Total Completed Reps/Hold: {proc.counter}")
            avg_a = int(np.mean(proc.accuracy_scores)) if proc.accuracy_scores else proc.last_acc
            c.drawString(40, 660, f"Avg Form Accuracy: {avg_a}%")
        c.save()
        return pdf_path

    if st.button("📥 Export PDF Audit Document"):
        pdf = generate_pdf_report()
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name="Biomechanics_Report.pdf", mime="application/pdf")