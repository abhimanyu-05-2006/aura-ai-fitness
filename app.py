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

# dotenv for reading hidden .env files
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
    
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F0F6FC !important;
        font-weight: 600 !important;
    }

    /* Input & Select Box styling in Sidebar */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: #21262d !important;
        color: #FFFFFF !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.8) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px !important;
        padding: 16px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    /* Status Badges */
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

    /* High-Contrast Tab Switcher */
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

    /* Chat message styling */
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

# Web Speech Synthesis Voice Guidance Engine
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
# 2. ADVANCED MULTI-EXERCISE BIOMECHANICS ENGINE
# ==============================================================================
class BiomechanicsEngine:
    def __init__(self, model_path="pose_landmarker.task"):
        if not os.path.exists(model_path):
            st.error(f"❌ Model file '{model_path}' not found! Download 'pose_landmarker.task' into the working directory.")
            st.stop()
        
        # Explicit CPU Delegate to prevent crashes on Linux environments
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
        self.reset_session()

    def reset_session(self):
        self.counter = 0
        self.stage = "UP"
        self.accuracy_scores = []
        self.snapshots = []
        self.rep_logs = []
        self.combo = 0
        self.max_combo = 0
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
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CLAHE Image Enhancement for Low-Light environments
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) < 70:
            lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            rgb_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        canvas_img = frame.copy()

        feedback = "Position yourself clearly in frame"
        warning = ""
        accuracy = 100
        voice_prompt = ""

        if not results.pose_landmarks:
            cv2.putText(canvas_img, "NO USER DETECTED IN CAMERA FOV", (int(w*0.15), int(h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return canvas_img, self.counter, self.stage, "No user detected", "Step into frame", 0, 0, self.combo, self.snapshots, self.boss_hp, 0, 0, ""

        landmarks = results.pose_landmarks[0]
        def get_pt(idx): return [landmarks[idx].x * w, landmarks[idx].y * h]

        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        l_shoulder, r_shoulder = get_pt(LEFT_SHOULDER), get_pt(RIGHT_SHOULDER)
        l_elbow, r_elbow = get_pt(LEFT_ELBOW), get_pt(RIGHT_ELBOW)
        l_wrist, r_wrist = get_pt(LEFT_WRIST), get_pt(RIGHT_WRIST)
        l_hip, r_hip = get_pt(LEFT_HIP), get_pt(RIGHT_HIP)
        l_knee, r_knee = get_pt(LEFT_KNEE), get_pt(RIGHT_KNEE)
        l_ankle, r_ankle = get_pt(LEFT_ANKLE), get_pt(RIGHT_ANKLE)

        shoulder_width = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
        scale_factor = 0.45 / (shoulder_width + 1e-6)

        if exercise == "Auto Detect":
            hip_avg_y = (l_hip[1] + r_hip[1]) / 2
            wrist_avg_y = (l_wrist[1] + r_wrist[1]) / 2
            if abs(l_shoulder[1] - l_hip[1]) < 0.25 * h:
                exercise = "Push-ups" if wrist_avg_y > hip_avg_y else "Plank"
            else:
                exercise = "Squats"

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

        # --- EXERCISE BIOMECHANICS EVALUATION ---
        if exercise == "Squats":
            accuracy = max(0, 100 - int(asymmetry_delta * 1.5))
            if l_knee[0] > l_ankle[0] + (30 / scale_factor * 0.001):
                warning += "Knee Shear Stress High! "
                accuracy -= 20
                voice_prompt = "Keep knees behind toes"

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
            if l_hip_angle <= 150:
                warning += "Hips Sagging!"
                voice_prompt = "Keep core tight and hips level"
            
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

        elif exercise in ["Plank", "Warrior II (Yoga)", "Tree Pose (Yoga)"]:
            self.jitter_buffer.append(l_hip)
            if len(self.jitter_buffer) > 10: self.jitter_buffer.pop(0)
            jitter = np.std(self.jitter_buffer) if len(self.jitter_buffer) > 1 else 0

            if not self.hold_start_time: self.hold_start_time = time.time()
            self.hold_duration = int(time.time() - self.hold_start_time)
            accuracy = max(0, 100 - int(jitter * 4))
            feedback = f"Holding Position... Tremor Index: {round(jitter, 1)}"
            
            if self.hold_duration % 10 == 0 and self.hold_duration > 0:
                voice_prompt = f"Great hold! {self.hold_duration} seconds"

        elif exercise == "Lunges":
            accuracy = max(0, 100 - int(asymmetry_delta * 1.2))
            if l_knee_angle < 90 or r_knee_angle < 90:
                self.stage = "DOWN"
                feedback = "Step back up to standing"
            if l_knee_angle > 160 and r_knee_angle > 160 and self.stage == "DOWN":
                self.stage = "UP"
                self.counter += 1
                self.accuracy_scores.append(accuracy)
                self.boss_hp = max(0, self.boss_hp - 10)
                voice_prompt = f"Lunge complete! Rep {self.counter}"
                if accuracy >= 90:
                    self.snapshots.append(canvas_img.copy())

        else:
            feedback = f"Executing {exercise}..."
            accuracy = 95

        if voice_prompt and (curr_time - self.last_speech_time > 4.0):
            self.last_speech_time = curr_time
        else:
            voice_prompt = ""

        # Draw Joint Overlays
        for joint in [l_knee, r_knee, l_elbow, r_elbow, l_hip, r_hip]:
            cv2.circle(canvas_img, (int(joint[0]), int(joint[1])), 7, (0, 255, 255), -1)

        avg_acc = int(np.mean(self.accuracy_scores)) if self.accuracy_scores else 100
        return canvas_img, self.counter, self.stage, feedback, warning, accuracy, avg_acc, self.combo, self.snapshots, self.boss_hp, self.velocity, self.power_watts, voice_prompt

if 'engine' not in st.session_state:
    st.session_state.engine = BiomechanicsEngine()

# ==============================================================================
# 3. FRONTEND DASHBOARD & DUAL AI ASSISTANT INTEGRATION
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

# SIDEBAR CONFIGURATION
st.sidebar.markdown("## 🎛️ Control Hub")

selected_exercise = st.sidebar.selectbox(
    "Select Target Exercise",
    ["Squats", "Push-ups", "Bicep Curls", "Lunges", "Plank", "Warrior II (Yoga)", "Tree Pose (Yoga)", "Auto Detect"]
)

user_weight_kg = st.sidebar.number_input("User Mass (kg)", min_value=30, max_value=200, value=70)

st.sidebar.markdown("---")
if gemini_api_key:
    st.sidebar.success("⚡ Live Gemini AI Engine Active")
else:
    st.sidebar.info("💡 Local Smart Rules Active (No Key Needed)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎙️ AI Voice Guidance")
enable_voice_coach = st.sidebar.toggle("Enable Voice Guidance", value=True)

camera_run = st.sidebar.checkbox('⚡ Initialize Camera Pipeline', value=False)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Telemetry Session", use_container_width=True):
    st.session_state.engine.reset_session()
    st.rerun()

# Dashboard Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 Live Biomechanics HUD", 
    "📊 Kinematics & Power", 
    "🤖 AI Fitness Assistant", 
    "📸 Flawless Rep Snapshots", 
    "📑 Clinical PDF Audit"
])

with tab1:
    col_cam, col_hud = st.columns([2.2, 1])
    with col_cam:
        FRAME_DISPLAY = st.image([])
    with col_hud:
        st.markdown("### 📈 Live Telemetry")
        m_reps = st.empty()
        m_acc = st.empty()
        badge_holder = st.empty()
        st.markdown("##### 👾 AI Boss Health")
        boss_bar = st.progress(100)
        box_feedback = st.empty()
        box_warning = st.empty()

with tab2:
    st.markdown("### ⚡ Velocity-Based Training (VBT) Metrics")
    col_v1, col_v2, col_v3 = st.columns(3)
    v_speed = col_v1.metric("Limb Velocity", "0.0 m/s")
    v_power = col_v2.metric("Mechanical Power Output", "0 Watts")
    v_cal = col_v3.metric("Mechanical Work Burned", "0 kcal")

# AI ASSISTANT TAB
with tab3:
    st.markdown("### 🤖 AURA AI Personal Coach & Assistant")
    st.write("Aapke Age, Weight, Height aur Fitness Goals ke basis par personalized suggestions aur exercise guide:")

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    age = col_a1.number_input("Age (Years)", min_value=10, max_value=100, value=22)
    height_cm = col_a2.number_input("Height (cm)", min_value=100, max_value=230, value=175)
    fitness_goal = col_a3.selectbox("Primary Fitness Goal", ["Weight Loss", "Muscle Gain", "Endurance & Flexibility", "Posture Correction"])
    activity_level = col_a4.selectbox("Activity Level", ["Beginner", "Intermediate", "Advanced"])

    height_m = height_cm / 100
    bmi = round(user_weight_kg / (height_m ** 2), 1)

    st.markdown("---")
    st.markdown("#### 📊 Profile Analysis Summary")
    st.info(f"**BMI:** {bmi} | **Target Goal:** {fitness_goal} | **Level:** {activity_level}")

    def get_ai_advice(age, weight, height, goal, level, exercise):
        advice = []
        if goal == "Weight Loss":
            advice.append("🔥 **Strategy:** High repetition sets (15-20 reps) with short rest intervals (30-45s) to maximize calorie burn.")
            advice.append("🥗 **Nutrition:** Aim for a 300-500 kcal caloric deficit with high protein intake (1.6g/kg).")
        elif goal == "Muscle Gain":
            advice.append("💪 **Strategy:** Focus on moderate reps (8-12 reps) with progressive overload and controlled eccentrics (3s down).")
            advice.append("🥗 **Nutrition:** Maintain a 250-400 kcal caloric surplus with 2.0g/kg protein intake.")
        elif goal == "Posture Correction":
            advice.append("🧘 **Strategy:** Prioritize isometric holds like Planks, Glute Bridges, and Yoga poses to strengthen core stabilization.")

        if age > 50:
            advice.append("⚠️ **Age Advisory:** Protect joint health by reducing impact velocity. Ensure thorough 10-min warmups.")
        if weight > 90 and exercise in ["Squats", "Lunges"]:
            advice.append("🦵 **Joint Guidance:** Focus on knee tracking; avoid letting knees cave inward under heavy loads.")

        return "\n\n".join(advice)

    st.markdown("#### 💡 Customized Training Guidelines")
    st.markdown(get_ai_advice(age, user_weight_kg, height_cm, fitness_goal, activity_level, selected_exercise))

    st.markdown("---")
    st.markdown("#### 💬 Ask AURA AI Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask any question about exercise form, diet, or workout plans:", placeholder="E.g., Squats karte waqt knee pain se kaise bache?")
    
    if st.button("Send Question") and user_query:
        if gemini_api_key and GEMINI_INSTALLED:
            try:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_prompt = f"""
                You are AURA AI, an elite fitness & biomechanics coach.
                User Context: Age {age}, Weight {user_weight_kg}kg, Height {height_cm}cm (BMI: {bmi}), Goal: {fitness_goal}, Current Exercise: {selected_exercise}.
                User Question: {user_query}
                Provide a clear, concise, professional response in simple Hinglish or English.
                """
                response_obj = model.generate_content(context_prompt)
                response = response_obj.text
            except Exception as e:
                response = f"⚠️ Gemini API Error: {str(e)}. Falling back to local smart engine."
        else:
            query_lower = user_query.lower()
            if "knee" in query_lower or "pain" in query_lower:
                response = "Knee pain avoid karne ke liye: 1) Knees ko toes ke aage zyadatar mat jaane do. 2) Squat karte waqt heels ground par rakho. 3) Quadriceps aur Hips ko proper warm up karo."
            elif "diet" in query_lower or "protein" in query_lower:
                response = f"Aapke weight ({user_weight_kg}kg) ke hisab se daily ~{int(user_weight_kg * 1.8)}g protein recommend hota hai. Eggs, Chicken, Paneer, Tofu, aur Dal include karein."
            elif "form" in query_lower or "posture" in query_lower:
                response = f"{selected_exercise} me proper posture ke liye spine ko neutral rakhein aur core muscles ko tight rakhein."
            else:
                response = f"Aapka query '{user_query}' recieve ho gaya hai. Optimum results ke liye daily 3-4 sets consistent repetitions ke sath complete karein."
            
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("ai", response))

    # Render Chat History
    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.markdown(f'<div class="user-msg"><b>👤 You:</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg"><b>🤖 AURA AI Assistant:</b> {msg}</div>', unsafe_allow_html=True)

with tab4:
    st.markdown("### 📸 Captured Flawless Form Snapshots (>90% Accuracy)")
    snaps = st.session_state.engine.snapshots
    if snaps:
        cols = st.columns(3)
        for idx, snap in enumerate(snaps[-6:]):
            cols[idx % 3].image(snap, channels="BGR", caption=f"Flawless Rep #{idx+1}")
    else:
        st.info("No >90% accuracy snapshots captured yet. Start performing reps with high accuracy!")

with tab5:
    st.markdown("### 📄 Generate Clinical Performance Report")
    
    def generate_pdf_report(engine):
        pdf_path = "Biomechanics_Audit_Report.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, 750, "AURA AI - Biomechanics & Posture Audit")
        c.setFont("Helvetica", 11)
        c.drawString(40, 725, f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(40, 710, f"Total Completed Reps/Hold: {engine.counter if selected_exercise not in ['Plank', 'Tree Pose (Yoga)', 'Warrior II (Yoga)'] else engine.hold_duration}")
        c.drawString(40, 695, f"Avg Form Accuracy: {int(np.mean(engine.accuracy_scores)) if engine.accuracy_scores else 100}%")
        c.save()
        return pdf_path

    if st.button("📥 Export PDF Audit Document"):
        pdf = generate_pdf_report(st.session_state.engine)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name="Biomechanics_Report.pdf", mime="application/pdf")

# ==============================================================================
# 4. CAMERA LOOP PIPELINE
# ==============================================================================
if camera_run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_run:
        ret, frame = cap.read()
        if not ret:
            st.error("Hardware Camera Stream Failure.")
            break
            
        frame = cv2.flip(frame, 1)
        
        processed_frame, reps, stage, feedback, warning, acc, avg_acc, combo, snaps, boss_hp, vel, power, voice_prompt = \
            st.session_state.engine.process_frame(frame, selected_exercise, user_weight_kg)

        FRAME_DISPLAY.image(processed_frame, channels="BGR", use_container_width=True)

        if voice_prompt:
            play_voice_guidance(voice_prompt, enable_voice=enable_voice_coach)

        is_hold_ex = selected_exercise in ["Plank", "Tree Pose (Yoga)", "Warrior II (Yoga)"]
        m_reps.metric("Completed Reps / Hold Time", f"{st.session_state.engine.hold_duration}s" if is_hold_ex else reps)
        m_acc.metric("Instant Form Accuracy", f"{acc}%", delta=f"Avg: {avg_acc}%")
        boss_bar.progress(boss_hp)
        
        v_speed.metric("Limb Velocity", f"{round(vel, 2)} m/s")
        v_power.metric("Mechanical Power Output", f"{round(power, 1)} W")
        
        mech_work_kcal = round((power * (time.time() - st.session_state.engine.prev_time)) * 0.000239006, 3)
        v_cal.metric("Mechanical Work Burned", f"{mech_work_kcal} kcal")

        if acc >= 85:
            badge_holder.markdown('<div class="status-badge badge-perfect">🟢 PERFECT BIOMECHANICAL FORM</div>', unsafe_allow_html=True)
        elif acc >= 60:
            badge_holder.markdown('<div class="status-badge badge-warn">🟡 MINOR POSTURE DEVIATION</div>', unsafe_allow_html=True)
        else:
            badge_holder.markdown('<div class="status-badge badge-danger">🔴 HIGH RISK INJURY CORRECTION</div>', unsafe_allow_html=True)

        box_feedback.info(f"💡 **AI Guidance**: {feedback}")
        if warning:
            box_warning.error(f"⚠️ **Hazard Alert**: {warning}")
        else:
            box_warning.empty()

    cap.release()