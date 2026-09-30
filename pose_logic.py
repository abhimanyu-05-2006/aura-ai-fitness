import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO
from utils import calculate_angle, compute_accuracy

# Voice Alerts Engine (Async threading to avoid frame lag)
class VoiceEngine:
    def __init__(self):
        self.last_spoken = ""
        self.last_time = 0

    def speak(self, text):
        current_time = time.time()
        if text != self.last_spoken or (current_time - self.last_time) > 3.0:
            self.last_spoken = text
            self.last_time = current_time
            threading.Thread(target=self._run_speech, args=(text,), daemon=True).start()

    def _run_speech(self, text):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass

voice = VoiceEngine()

class PoseEvaluator:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        self.reset_stats()

    def reset_stats(self):
        self.counter = 0
        self.stage = None
        self.combo = 0
        self.accuracy_scores = []
        self.snapshots = []
        self.tree_hold_start = None
        self.tree_hold_time = 0

    def process_frame(self, image, exercise="Squat"):
        results = self.model(image, verbose=False)
        feedback = "Position yourself in camera"
        injury_warning = None
        accuracy = 0.0
        annotated_frame = results[0].plot()

        if len(results[0].keypoints) > 0 and results[0].keypoints.xy.shape[1] > 0:
            kp = results[0].keypoints.xy[0].cpu().numpy()

            if len(kp) >= 16:
                # Keypoints index mapping
                l_shoulder, l_elbow, l_wrist = kp[5], kp[7], kp[9]
                l_hip, l_knee, l_ankle = kp[11], kp[13], kp[15]

                if exercise == "Squat":
                    if l_hip[0] > 0 and l_knee[0] > 0 and l_ankle[0] > 0:
                        angle = calculate_angle(l_hip, l_knee, l_ankle)
                        accuracy = compute_accuracy(angle, 90) # Target 90 deg depth

                        # Injury Warning: Extreme knee overextension/collapse
                        if angle < 60:
                            injury_warning = "🚨 DANGER: Squatting Too Deep! Protect Knee Joints."
                            voice.speak("Danger, squatting too deep")

                        if angle > 160:
                            self.stage = "up"
                            feedback = "Go Down"
                        if angle < 100 and self.stage == "up":
                            self.stage = "down"
                            self.counter += 1
                            self.combo += 1
                            feedback = f"Good Rep! Reps: {self.counter}"
                            voice.speak("Good Rep")
                            
                            # Auto-Snap Best Rep
                            if accuracy > 85:
                                self.snapshots.append(image.copy())

                        self.accuracy_scores.append(accuracy)

                elif exercise == "Bicep Curl":
                    if l_shoulder[0] > 0 and l_elbow[0] > 0 and l_wrist[0] > 0:
                        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                        accuracy = compute_accuracy(angle, 40)

                        if angle > 160:
                            self.stage = "down"
                            feedback = "Curl Up"
                        if angle < 40 and self.stage == "down":
                            self.stage = "up"
                            self.counter += 1
                            self.combo += 1
                            feedback = "Good Rep!"
                            voice.speak("Good Rep")
                            if accuracy > 85:
                                self.snapshots.append(image.copy())

                        self.accuracy_scores.append(accuracy)

                elif exercise == "Push-ups":
                    if l_shoulder[0] > 0 and l_elbow[0] > 0 and l_wrist[0] > 0:
                        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                        accuracy = compute_accuracy(angle, 70)

                        if angle > 150:
                            self.stage = "up"
                            feedback = "Lower Chest"
                        if angle < 80 and self.stage == "up":
                            self.stage = "down"
                            self.counter += 1
                            self.combo += 1
                            feedback = "Good Push-up!"
                            voice.speak("Good Push up")

                        self.accuracy_scores.append(accuracy)

                elif exercise == "Plank":
                    # Shoulder-Hip-Ankle alignment
                    if l_shoulder[0] > 0 and l_hip[0] > 0 and l_ankle[0] > 0:
                        angle = calculate_angle(l_shoulder, l_hip, l_ankle)
                        accuracy = compute_accuracy(angle, 180) # Straight line

                        if angle < 155:
                            injury_warning = "🚨 INJURY RISK: Lower back dipping! Tighten Core."
                            feedback = "Keep Body Straight!"
                            voice.speak("Keep body straight")
                        else:
                            feedback = "Perfect Plank Form!"
                        self.accuracy_scores.append(accuracy)

                elif exercise == "Tree Pose (Yoga)":
                    if l_hip[0] > 0 and l_knee[0] > 0 and l_ankle[0] > 0:
                        angle = calculate_angle(l_hip, l_knee, l_ankle)
                        accuracy = compute_accuracy(angle, 45) # Raised leg angle

                        if accuracy > 70:
                            if self.tree_hold_start is None:
                                self.tree_hold_start = time.time()
                            self.tree_hold_time = int(time.time() - self.tree_hold_start)
                            feedback = f"Holding Pose: {self.tree_hold_time}s 🔥"
                            if self.tree_hold_time % 5 == 0 and self.tree_hold_time > 0:
                                voice.speak(f"Holding for {self.tree_hold_time} seconds")
                        else:
                            self.tree_hold_start = None
                            feedback = "Balance on one leg"

        avg_acc = round(sum(self.accuracy_scores)/len(self.accuracy_scores), 1) if self.accuracy_scores else 0.0
        return annotated_frame, self.counter, self.stage, feedback, injury_warning, accuracy, avg_acc, self.combo, self.snapshots