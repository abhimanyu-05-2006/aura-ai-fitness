import os
import time
import cv2
import requests
import streamlit as st
import streamlit.components.v1 as components
from pose_logic import PoseEvaluator
from utils import generate_pdf_report

# Page Configuration
st.set_page_config(
    page_title="AURA Fitness | AI Biomechanics & Pose Analytics", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ==========================================
# ADVANCED UI/UX STYLING (GLASSMORPHISM ENGINE)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Global Dark Cyber Theme */
    .stApp {
        background: radial-gradient(circle at top right, #111827, #030712);
        color: #F9FAFB;
    }

    /* Glassmorphic Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px !important;
        padding: 18px 22px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Custom Glass Container */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }

    /* Animated Glowing Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 18px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
    }

    .badge-excellent {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        animation: pulse-green 2s infinite;
    }

    .badge-good {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }

    .badge-poor {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid #EF4444;
        animation: pulse-red 1.5s infinite;
    }

    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Tab Switcher Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(17, 24, 39, 0.5);
        padding: 6px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        color: #9CA3AF;
        font-weight: 600;
        border: none !important;
        padding: 0 24px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F17;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Custom Streamlit Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Web Speech API Natural Voice Feedback Engine
def trigger_voice_guidance(phrase):
    js_code = f"""
    <script>
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel(); // Clear queue
        var msg = new SpeechSynthesisUtterance("{phrase}");
        msg.rate = 1.0;
        msg.pitch = 1.0;
        msg.lang = 'en-US';
        window.speechSynthesis.speak(msg);
    }}
    </script>
    """
    components.html(js_code, height=0, width=0)

# Environment Variables & State
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = PoseEvaluator()
if 'last_speech_time' not in st.session_state:
    st.session_state.last_speech_time = 0

# Main Header
st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0 25px 0;">
        <div>
            <h1 style="margin: 0; font-weight: 800; font-size: 2.2rem; background: linear-gradient(135deg, #FFF 0%, #9CA3AF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ AURA AI <span style="font-size:1.2rem; font-weight:400; color:#6366F1;">Pose & Biomechanics Engine</span>
            </h1>
            <p style="margin: 5px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
                Real-time joint tracking, posture estimation, and exercise form evaluation.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Control Hub
st.sidebar.markdown("### 🎛️ Control Center")
exercise = st.sidebar.selectbox("Target Exercise Mode", ["Squats", "Bicep Curl", "Push-ups", "Plank", "Tree Pose (Yoga)"])
voice_coaching = st.sidebar.toggle("🎙️ Enable AI Voice Feedback", value=False)
run = st.sidebar.checkbox('🎥 Start Vision Feed', value=False)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Workout Metrics", use_container_width=True):
    st.session_state.evaluator.reset_stats()
    st.session_state.start_time = time.time()
    st.sidebar.success("Metrics successfully reset.")

# Grok API Assistant Engine
def get_ai_response(user_prompt, key=GROK_API_KEY):
    if key:
        try:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": "grok-beta",
                "messages": [
                    {"role": "system", "content": "You are an elite AI Biomechanics Coach."},
                    {"role": "user", "content": user_prompt}
                ]
            }
            res = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers, timeout=5)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content']
        except Exception:
            pass
    
    prompt_lower = user_prompt.lower()
    if "squat" in prompt_lower:
        return "💡 **Biomechanics Insight**: Ensure knees track in line with toes. Maintain a neutral spine and aim for hips dropping parallel to the knee joint."
    elif "plank" in prompt_lower:
        return "💡 **Biomechanics Insight**: Avoid hip sag. Engage the core complex by pulling the navel toward the spine."
    else:
        return "🤖 **AURA Engine**: Focus on movement tempo control and target an accuracy rating above 80%."

# Layout
tab1, tab2, tab3 = st.tabs(["🎥 Live Vision Evaluator", "📸 Rep Snapshots", "🤖 AI Coach Assistant"])

with tab1:
    col_cam, col_metrics = st.columns([2.3, 1])
    
    with col_cam:
        FRAME_WINDOW = st.image([])
    
    with col_metrics:
        st.markdown("### 📈 Live Telemetry")
        counter_metric = st.empty()
        accuracy_metric = st.empty()
        accuracy_badge = st.empty()
        combo_metric = st.empty()
        feedback_box = st.empty()
        warning_box = st.empty()
        
        elapsed = int(time.time() - st.session_state.start_time)
        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #9CA3AF; font-size: 0.85rem;">Session Duration</span>
                    <span style="font-weight: 600; font-size: 0.95rem;">{elapsed}s</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #9CA3AF; font-size: 0.85rem;">Est. Energy Burned</span>
                    <span style="font-weight: 600; color: #F59E0B; font-size: 0.95rem;">{round(elapsed * 0.12, 1)} kcal</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("📸 High-Accuracy Form Snapshots (>90% Score)")
    snapshots_container = st.container()

with tab3:
    st.subheader("💬 AI Biomechanics Consultant")
    user_q = st.text_input("Inquire about movement mechanics, joint strain, or nutrition:")
    if user_q:
        ans = get_ai_response(user_q)
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #6366F1;">
                {ans}
            </div>
        """, unsafe_allow_html=True)

# Vision Stream Loop
if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera feed disconnected or unavailable.")
            break
            
        frame = cv2.flip(frame, 1)
        
        # Pose Engine Processing
        processed_frame, reps, stage, feedback, warning, cur_acc, avg_acc, combo_val, snaps = st.session_state.evaluator.process_frame(frame, exercise)
        
        FRAME_WINDOW.image(processed_frame, channels="BGR", use_container_width=True)
        
        # Metric Updates
        counter_metric.metric("Total Completed Reps", reps if exercise != "Tree Pose (Yoga)" else f"{st.session_state.evaluator.tree_hold_time}s")
        accuracy_metric.metric("Current Form Score", f"{cur_acc}%", delta=f"Session Avg: {avg_acc}%")
        
        # Form Quality State
        if cur_acc >= 85:
            accuracy_badge.markdown('<div class="badge badge-excellent">🟢 EXCELLENT FORM</div>', unsafe_allow_html=True)
        elif cur_acc >= 65:
            accuracy_badge.markdown('<div class="badge badge-good">🟡 ACCEPTABLE FORM</div>', unsafe_allow_html=True)
        else:
            accuracy_badge.markdown('<div class="badge badge-poor">🔴 FORM CORRECTION NEEDED</div>', unsafe_allow_html=True)
            
            # Smart AI Voice Feedback (Triggers once every 5 seconds to prevent overlap)
            if voice_coaching and (time.time() - st.session_state.last_speech_time > 5.0):
                trigger_voice_guidance("Please check your form and adjust posture")
                st.session_state.last_speech_time = time.time()

        if combo_val >= 3:
            combo_metric.markdown(f"""
                <div style="background: linear-gradient(90deg, #F59E0B, #EF4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.1rem; padding: 10px 0;">
                    🔥 {combo_val}x PERFECT STREAK COMBO!
                </div>
            """, unsafe_allow_html=True)
        else:
            combo_metric.empty()

        feedback_box.info(f"💡 **Guidance**: {feedback}")
        
        if warning:
            warning_box.error(warning)
        else:
            warning_box.empty()
            
        # Snapshot Gallery Rendering
        if snaps:
            with snapshots_container:
                cols = st.columns(3)
                for idx, snap in enumerate(snaps[-6:]):
                    cols[idx % 3].image(snap, channels="BGR", caption=f"Capture #{idx+1}")

    cap.release()

# Sidebar Export Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Export Analytics")
if st.sidebar.button("Generate Performance PDF", use_container_width=True):
    tot_time = int(time.time() - st.session_state.start_time)
    eval_obj = st.session_state.evaluator
    avg_acc = round(sum(eval_obj.accuracy_scores)/len(eval_obj.accuracy_scores), 1) if eval_obj.accuracy_scores else 0.0
    
    data = {
        "duration": tot_time,
        "reps": eval_obj.counter,
        "avg_accuracy": avg_acc,
        "calories": round(tot_time * 0.12, 1)
    }
    
    pdf_file = generate_pdf_report(data)
    with open(pdf_file, "rb") as f:
        st.sidebar.download_button("📥 Download Report", f, file_name="AURA_Workout_Report.pdf", mime="application/pdf", use_container_width=True)