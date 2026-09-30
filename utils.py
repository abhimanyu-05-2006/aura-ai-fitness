import numpy as np
from fpdf import FPDF
import datetime

def calculate_angle(a, b, c):
    """3 points ke beech ka joint angle compute karta hai"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def compute_accuracy(actual_angle, target_angle):
    """Dynamic Form Accuracy Score (0-100%) Calculator"""
    deviation = abs(target_angle - actual_angle)
    score = max(0, 100 - (deviation * 1.5))
    return round(score, 1)

def generate_pdf_report(session_data):
    """Auto-Generated PDF Workout Audit Report Generator"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 10, "AI Fitness & Pose Audit Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, f"Total Workout Duration: {session_data.get('duration', '0')} sec", ln=True)
    pdf.cell(200, 10, f"Total Reps Completed: {session_data.get('reps', 0)}", ln=True)
    pdf.cell(200, 10, f"Average Form Accuracy: {session_data.get('avg_accuracy', 0)}%", ln=True)
    pdf.cell(200, 10, f"Estimated Calories Burned: {session_data.get('calories', 0)} kcal", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Improvement Feedback:", ln=True)
    pdf.set_font("Arial", size=12)
    
    avg_acc = session_data.get('avg_accuracy', 0)
    if avg_acc >= 85:
        feedback = "Excellent form! Maintain this posture for optimal muscle targeting."
    elif avg_acc >= 65:
        feedback = "Good performance! Keep an eye on depth and back posture during reps."
    else:
        feedback = "Focus on slower execution. Pay attention to injury warnings."
        
    pdf.multi_cell(0, 10, feedback)
    filename = "workout_report.pdf"
    pdf.output(filename)
    return filename