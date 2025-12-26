import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
import os
from tqdm import tqdm
import statistics

# --- Data Classes ---
@dataclass
class NoteEvent:
    """Represents a single musical note event."""
    time: float
    note: str
    velocity: int
    duration: float
    hand: Optional[str] = None
    finger: Optional[int] = None
    confidence: Optional[float] = None

@dataclass
class PianoKey:
    """Represents the position and color of a piano key."""
    name: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    baseline_color: Optional[np.ndarray] = None

@dataclass
class PerformanceMetrics:
    """Contains calculated performance metrics including BPM."""
    bpm: float
    total_notes: int
    performance_duration: float
    average_note_interval: float
    tempo_consistency: float

# --- Main Analysis Class ---
class PianoPerformanceAnalyzer:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
        self.video_path = video_path
        self.piano_keys: Dict[str, PianoKey] = {}
        self.time_offset = 0.0
        self.performance_metrics: Optional[PerformanceMetrics] = None

        # Expanded MIDI mapping for a wider range of notes
        self.note_to_midi = {
            'C3': 48, 'D3': 50, 'E3': 52, 'F3': 53, 'G3': 55, 'A3': 57, 'B3': 59,
            'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71,
            'C5': 72, 'D5': 74, 'E5': 76, 'F5': 77, 'G5': 79, 'A5': 81, 'B5': 83
        }

        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.6, min_tracking_confidence=0.5
        )

    def calculate_performance_metrics(self, notes: List[NoteEvent]) -> PerformanceMetrics:
        """Calculate BPM and other performance metrics from the note data."""
        if len(notes) < 2:
            return PerformanceMetrics(0, len(notes), 0, 0, 0)
        
        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda x: x.time)
        
        # Calculate basic metrics
        total_notes = len(notes)
        performance_duration = sorted_notes[-1].time - sorted_notes[0].time
        
        # Calculate intervals between consecutive notes
        intervals = []
        for i in range(1, len(sorted_notes)):
            interval = sorted_notes[i].time - sorted_notes[i-1].time
            intervals.append(interval)
        
        # Filter out very long pauses (likely rests) for BPM calculation
        # Keep intervals under 2 seconds for tempo calculation
        tempo_intervals = [interval for interval in intervals if interval < 2.0 and interval > 0.1]
        
        if not tempo_intervals:
            tempo_intervals = intervals
        
        average_interval = statistics.mean(tempo_intervals) if tempo_intervals else 0
        
        # Calculate BPM - assuming quarter notes for now
        # BPM = 60 / (average time between beats in seconds)
        bpm = 60 / average_interval if average_interval > 0 else 0
        
        # Calculate tempo consistency (lower standard deviation = more consistent)
        tempo_consistency = 1 / (statistics.stdev(tempo_intervals) + 0.001) if len(tempo_intervals) > 1 else 1
        
        return PerformanceMetrics(
            bpm=round(bpm, 1),
            total_notes=total_notes,
            performance_duration=round(performance_duration, 2),
            average_note_interval=round(average_interval, 3),
            tempo_consistency=round(tempo_consistency, 2)
        )

    def setup_piano_keys(self, keys_to_define: List[str], setup_frame_time: float = 1.0):
        """Interactively define bounding boxes for a specific list of piano keys."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(setup_frame_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read setup frame.")
            cap.release()
            return

        print("\n=== PIANO KEY SETUP ===")
        print("Click two opposite corners for each required key.")
        print("Controls: 'r' to reset, 'q' to quit, 's' to save and continue.")
        
        current_key_idx, temp_points = 0, []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_key_idx, temp_points
            if event == cv2.EVENT_LBUTTONDOWN and current_key_idx < len(keys_to_define):
                temp_points.append((x, y))
                if len(temp_points) == 2:
                    x_min, y_min = np.min(temp_points, axis=0)
                    x_max, y_max = np.max(temp_points, axis=0)
                    key_name = keys_to_define[current_key_idx]
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    # Store key with its baseline color
                    key_region = frame[y_min:y_max, x_min:x_max]
                    baseline_color = np.mean(key_region, axis=(0, 1))
                    self.piano_keys[key_name] = PianoKey(name=key_name, bbox=bbox, baseline_color=baseline_color)
                    print(f"-> Saved key: {key_name}")
                    
                    current_key_idx += 1
                    temp_points = []
        
        cv2.namedWindow('Piano Key Setup', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Piano Key Setup', mouse_callback)
        while True:
            display_frame = frame.copy()
            # Draw already defined keys
            for key_data in self.piano_keys.values():
                x, y, w, h = key_data.bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, key_data.name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display current instructions
            if current_key_idx < len(keys_to_define):
                key_to_set = keys_to_define[current_key_idx]
                instruction = f"Define '{key_to_set}' ({current_key_idx + 1}/{len(keys_to_define)}). Click {len(temp_points) + 1}/2"
                cv2.putText(display_frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
            else:
                cv2.putText(display_frame, "All keys defined. Press 's' to continue.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Piano Key Setup', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): break
            elif key == ord('r'):
                self.piano_keys.clear(); current_key_idx = 0; temp_points = []
                print("Reset all key definitions.")

        cv2.destroyAllWindows()
        cap.release()
        self.save_key_config()

    def save_key_config(self, filename: str = "piano_keys.txt"):
        """Saves the defined piano key bounding boxes and colors to a file."""
        with open(filename, 'w') as f:
            for key_name, key_data in self.piano_keys.items():
                x, y, w, h = key_data.bbox
                r, g, b = key_data.baseline_color
                f.write(f"{key_name},{x},{y},{w},{h},{r:.2f},{g:.2f},{b:.2f}\n")
        print(f"Saved {len(self.piano_keys)} keys to {filename}")

    def load_key_config(self, filename: str = "piano_keys.txt") -> bool:
        """Loads piano key configuration from a file."""
        if not os.path.exists(filename): return False
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    key_name = parts[0]
                    bbox = tuple(map(int, parts[1:5]))
                    baseline_color = np.array(list(map(float, parts[5:8])))
                    self.piano_keys[key_name] = PianoKey(name=key_name, bbox=bbox, baseline_color=baseline_color)
            print(f"Loaded {len(self.piano_keys)} piano keys from {filename}")
            return True
        except Exception as e:
            print(f"Error loading key config: {e}")
            return False

    def parse_note_data(self, note_text: str) -> List[NoteEvent]:
        """Parses the raw text of note data into a list of NoteEvent objects."""
        notes = []
        # Regex updated to make "Play" optional
        pattern = re.compile(r"(\d+\.\d+)\s+(?:Play\s+)?([A-G]#?\d)\s+\(V:(\d+),\s+D:([\d\.]+)s\)")
        for line in note_text.strip().split('\n'):
            match = pattern.search(line.strip())
            if match:
                time, note, velocity, duration = match.groups()
                notes.append(NoteEvent(time=float(time), note=note, velocity=int(velocity), duration=float(duration)))
        return sorted(notes, key=lambda x: x.time)

    def interactive_sync_tuning(self, notes: List[NoteEvent]) -> float:
        """Opens an interactive window to let the user manually sync video and notes."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_offset, frame_idx = self.time_offset, 0
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            current_video_time = frame_idx / fps
            
            # Draw instructions on the frame
            cv2.putText(frame, "Goal: Align BLUE flash with the key press", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Time: {current_video_time:.2f}s | Offset: {current_offset:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Controls: 'A'/'D' to Adjust, 'Q' to Accept", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Highlight keys that should be playing at this moment
            for note in notes:
                if abs(current_video_time - (note.time + current_offset)) < (1.0 / fps) and note.note in self.piano_keys:
                    x, y, w, h = self.piano_keys[note.note].bbox
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 100, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            cv2.imshow("Interactive Sync Tuning", frame)
            
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'): break
            elif key == ord('d'): current_offset += 0.05
            elif key == ord('a'): current_offset -= 0.05
            
            frame_idx += 1
            
        cv2.destroyAllWindows()
        cap.release()
        print(f"Interactive tuning complete. Final offset: {current_offset:.2f}s")
        return current_offset

    def assign_hands_and_fingers(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Automatically assigns hand and finger to each note using MediaPipe."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        finger_tip_ids = {
            1: self.mp_hands.HandLandmark.THUMB_TIP, 2: self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            3: self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 4: self.mp_hands.HandLandmark.RING_FINGER_TIP,
            5: self.mp_hands.HandLandmark.PINKY_TIP
        }

        for note in tqdm(notes, desc="Assigning Hands/Fingers"):
            corrected_time = note.time + self.time_offset
            frame_number = int(corrected_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            # Fallback if frame is invalid or key wasn't defined
            if not ret or note.note not in self.piano_keys:
                midi_note = self.note_to_midi.get(note.note, 60)
                # FLIPPED LOGIC: Lower notes -> right hand (player's perspective)
                note.hand = 'right' if midi_note < 60 else 'left'
                note.finger, note.confidence = 2, 0.2
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            x, y, w, h = self.piano_keys[note.note].bbox
            key_center = (x + w // 2, y + h // 2)
            min_dist, best_hand, best_finger = float('inf'), None, None

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    for finger_num, finger_tip_id in finger_tip_ids.items():
                        lm = hand_landmarks.landmark[finger_tip_id]
                        lm_px = (int(lm.x * frame_width), int(lm.y * frame_height))
                        dist = np.linalg.norm(np.array(key_center) - np.array(lm_px))
                        
                        if dist < min_dist:
                            min_dist = dist
                            # FLIPPED LOGIC: Swap Left and Right from MediaPipe
                            detected_hand = handedness.classification[0].label.lower()
                            best_hand = 'right' if detected_hand == 'left' else 'left'
                            best_finger = finger_num

            # Assign if a fingertip was found close enough
            if best_hand and min_dist < (w * 1.5): # Use a generous distance threshold
                note.hand, note.finger = best_hand, best_finger
                note.confidence = max(0.0, 1.0 - (min_dist / (w * 1.5)))
            else: # Fallback if no hands are detected nearby
                midi_note = self.note_to_midi.get(note.note, 60)
                # FLIPPED LOGIC: Lower notes -> right hand (player's perspective)
                note.hand = 'right' if midi_note < 60 else 'left'
                note.finger, note.confidence = 2, 0.3

        cap.release()
        return notes

    def interactive_finger_correction(self, analyzed_notes: List[NoteEvent]) -> List[NoteEvent]:
        """Allows the user to manually correct the automated finger/hand assignments."""
        print("\n=== Interactive Correction Session ===")
        print("Review each note. Press keys to correct, then Enter to confirm.")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        corrected_notes, quit_session = list(analyzed_notes), False
        i = 0
        while i < len(corrected_notes):
            note = corrected_notes[i]
            if quit_session: break

            # Seek to the correct frame
            corrected_time = note.time + self.time_offset
            frame_number = int(corrected_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret or note.note not in self.piano_keys:
                i += 1
                continue

            display_frame = frame.copy()
            x, y, w, h = self.piano_keys[note.note].bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Display info text
            progress_text = f"Note {i+1}/{len(corrected_notes)}: {note.note} @ {note.time:.2f}s"
            current_sel_text = f"Current -> Hand: {note.hand.upper()}, Finger: {note.finger}"
            controls_text = "L/R=Hand | 1-5=Finger | Enter=Next | B=Back | Q=Quit"
            
            cv2.putText(display_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2)
            cv2.putText(display_frame, current_sel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, controls_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Interactive Correction", display_frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13 or key == ord(' '):  # Enter or Space
                print(f"  -> Confirmed Note {i+1}: Hand={note.hand}, Finger={note.finger}")
                i += 1
            elif key == ord('q'): quit_session = True
            elif key == ord('b'): i = max(0, i - 1) # Go back one note
            elif key == ord('l'): note.hand = 'left'
            elif key == ord('r'): note.hand = 'right'
            elif ord('1') <= key <= ord('5'): note.finger = int(chr(key))
        
        cv2.destroyAllWindows()
        cap.release()
        return corrected_notes

    def full_analysis(self, note_text: str, interactive_sync: bool = True, interactive_correction: bool = True):
        """Runs the entire analysis pipeline from start to finish."""
        print("=== PIANO PERFORMANCE ANALYSIS PIPELINE ===")
        
        notes = self.parse_note_data(note_text)
        if not notes:
            raise ValueError("Could not parse any notes from the provided text.")

        # Calculate performance metrics including BPM
        self.performance_metrics = self.calculate_performance_metrics(notes)

        # Determine which keys need to be defined
        required_keys = sorted(list(set(n.note for n in notes)))
        print(f"Found {len(required_keys)} unique notes in the data.")
        
        self.load_key_config()
        missing_keys = [key for key in required_keys if key not in self.piano_keys]

        if missing_keys:
            print(f"Configuration is missing {len(missing_keys)} keys. Starting setup for them...")
            self.setup_piano_keys(missing_keys)

        if interactive_sync:
            print("Starting interactive synchronization...")
            self.time_offset = self.interactive_sync_tuning(notes)

        analyzed_notes = self.assign_hands_and_fingers(notes)
        
        if interactive_correction:
            analyzed_notes = self.interactive_finger_correction(analyzed_notes)
            
        print("\nAnalysis complete!")
        return analyzed_notes

    def print_results(self, analyzed_notes: List[NoteEvent]):
        """Prints the final analysis results in a clean, formatted table."""
        print("\n" + "=" * 80)
        print("🎹 PIANO PERFORMANCE ANALYSIS RESULTS 🎹")
        
        # Print performance metrics first
        if self.performance_metrics:
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  • BPM (Beats Per Minute): {self.performance_metrics.bpm}")
            print(f"  • Total Notes: {self.performance_metrics.total_notes}")
            print(f"  • Performance Duration: {self.performance_metrics.performance_duration}s")
            print(f"  • Average Note Interval: {self.performance_metrics.average_note_interval}s")
            print(f"  • Tempo Consistency Score: {self.performance_metrics.tempo_consistency}")
        
        print(f"\n📝 DETAILED NOTE ANALYSIS:")
        header = f"{'Time (s)':<10} {'Note':<8} {'Hand':<10} {'Finger':<15} {'Confidence':<12}"
        print(header)
        print("-" * len(header))
        for note in analyzed_notes:
            conf_str = f"{note.confidence:.2f}" if note.confidence is not None else "N/A"
            finger_map = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Ring", 5: "Pinky"}
            finger_str = f"{note.finger} ({finger_map.get(note.finger, '?')})" if note.finger else "N/A"
            hand_str = str(note.hand).capitalize() if note.hand else "N/A"
            print(f"{note.time:<10.2f} {note.note:<8} {hand_str:<10} {finger_str:<15} {conf_str:<12}")
        
    def save_results_to_csv(self, analyzed_notes, filename: str = "analysis_results.csv"):
        """Saves the detailed analysis results to a CSV file."""
        with open(filename, 'w', newline='') as f:
            # Add BPM as the first line if available
            if self.performance_metrics:
                f.write(f"# BPM: {self.performance_metrics.bpm}\n")
                f.write(f"# Total Notes: {self.performance_metrics.total_notes}\n")
                f.write(f"# Duration: {self.performance_metrics.performance_duration}s\n")
            
            f.write("timestamp,note,duration,velocity,hand,finger,confidence\n")
            for note in analyzed_notes:
                f.write(f"{note.time:.3f},{note.note},{note.duration:.3f},{note.velocity},"
                        f"{note.hand},{note.finger},{note.confidence:.3f}\n")
        print(f"\n✅ CSV data saved to: {filename}")

    def save_results_to_txt(self, analyzed_notes: List[NoteEvent], filename: str = "analysis_results.txt"):
        """Saves the detailed analysis results to a TXT file with full formatting."""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🎹 PIANO PERFORMANCE ANALYSIS RESULTS 🎹\n")
            f.write("=" * 80 + "\n\n")
            
            # Write performance metrics
            if self.performance_metrics:
                f.write("📊 PERFORMANCE METRICS:\n")
                f.write(f"  • BPM (Beats Per Minute): {self.performance_metrics.bpm}\n")
                f.write(f"  • Total Notes: {self.performance_metrics.total_notes}\n")
                f.write(f"  • Performance Duration: {self.performance_metrics.performance_duration}s\n")
                f.write(f"  • Average Note Interval: {self.performance_metrics.average_note_interval}s\n")
                f.write(f"  • Tempo Consistency Score: {self.performance_metrics.tempo_consistency}\n\n")
            
            f.write("📝 DETAILED NOTE ANALYSIS:\n")
            header = f"{'Time (s)':<10} {'Note':<8} {'Hand':<10} {'Finger':<15} {'Confidence':<12}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for note in analyzed_notes:
                conf_str = f"{note.confidence:.2f}" if note.confidence is not None else "N/A"
                finger_map = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Ring", 5: "Pinky"}
                finger_str = f"{note.finger} ({finger_map.get(note.finger, '?')})" if note.finger else "N/A"
                hand_str = str(note.hand).capitalize() if note.hand else "N/A"
                f.write(f"{note.time:<10.2f} {note.note:<8} {hand_str:<10} {finger_str:<15} {conf_str:<12}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Analysis completed at {os.path.basename(self.video_path)}\n")
            f.write("Generated by Piano Performance Analyzer\n")
            
        print(f"✅ TXT report saved to: {filename}")

# --- Example Usage ---
if __name__ == "__main__":
    note_data = """
3.82      Play C3 (V:100, D:1.27s)
3.83      Play C4 (V:100, D:0.28s)
4.43      Play C4 (V:100, D:0.27s)
5.03      Play E3 (V:100, D:1.20s)
5.05      Play G4 (V:100, D:0.22s)
5.61      Play G4 (V:100, D:0.24s)
6.18      Play A4 (V:100, D:0.20s)
6.20      Play F3 (V:100, D:1.15s)
6.75      Play A4 (V:100, D:0.20s)
7.30      Play G4 (V:100, D:0.72s)
7.31      Play E3 (V:100, D:1.23s)
8.50      Play F4 (V:100, D:0.20s)
8.51      Play D3 (V:100, D:1.20s)
9.11      Play F4 (V:100, D:0.22s)
9.69      Play E4 (V:100, D:0.19s)
9.70      Play C3 (V:100, D:1.17s)
10.26     Play E4 (V:100, D:0.21s)
10.85     Play G3 (V:100, D:1.19s)
10.88     Play D4 (V:100, D:0.17s)
11.42     Play D4 (V:100, D:0.17s)
12.02     Play C3 (V:100, D:1.00s)
12.02     Play C4 (V:100, D:0.87s)
13.27     Play G4 (V:100, D:0.28s)
13.28     Play C3 (V:100, D:1.22s)
13.87     Play G4 (V:100, D:0.27s)
14.45     Play F3 (V:100, D:1.20s)
14.46     Play F4 (V:100, D:0.18s)
15.05     Play F4 (V:100, D:0.22s)
15.64     Play C3 (V:100, D:1.21s)
15.65     Play E4 (V:100, D:0.15s)
16.22     Play E4 (V:100, D:0.17s)
16.83     Play G3 (V:100, D:0.94s)
16.84     Play D4 (V:100, D:0.84s)
18.06     Play C3 (V:100, D:1.19s)
18.09     Play G4 (V:100, D:0.21s)
18.62     Play G4 (V:100, D:0.21s)
19.20     Play F3 (V:100, D:1.18s)
19.23     Play F4 (V:100, D:0.19s)
19.79     Play F4 (V:100, D:0.19s)
20.37     Play C3 (V:100, D:1.04s)
20.39     Play E4 (V:100, D:0.14s)
20.97     Play E4 (V:100, D:0.31s)
21.52     Play G3 (V:100, D:0.78s)
21.54     Play D4 (V:100, D:0.76s)
22.74     Play C3 (V:100, D:1.21s)
22.75     Play C4 (V:100, D:0.18s)
23.33     Play C4 (V:100, D:0.19s)
23.90     Play E3 (V:100, D:1.21s)
23.93     Play G4 (V:100, D:0.18s)
24.50     Play G4 (V:100, D:0.18s)
25.08     Play F3 (V:100, D:1.22s)
25.11     Play A4 (V:100, D:0.17s)
25.67     Play A4 (V:100, D:0.17s)
26.27     Play C3 (V:100, D:0.90s)
26.29     Play G4 (V:100, D:0.80s)
27.49     Play D3 (V:100, D:1.22s)
27.53     Play F4 (V:100, D:0.24s)
28.12     Play F4 (V:100, D:0.21s)
28.71     Play C3 (V:100, D:1.02s)
28.71     Play E4 (V:100, D:0.17s)
29.35     Play E4 (V:100, D:0.20s)
29.97     Play G3 (V:100, D:0.90s)
29.98     Play D4 (V:100, D:0.22s)
30.61     Play D4 (V:100, D:0.21s)
31.22     Play C3 (V:100, D:1.52s)
31.24     Play C4 (V:100, D:1.54s)
    """
    video_file = "twinkle_simple.mp4" 

    try:
        # Initialize the analyzer with the video file
        analyzer = PianoPerformanceAnalyzer(video_file)
        
        # Run the full analysis pipeline
        # You can disable interactive steps for batch processing if needed
        results = analyzer.full_analysis(
            note_data, 
            interactive_sync=True, 
            interactive_correction=True
        )
        
        # Print the results to the console and save to files
        analyzer.print_results(results)
        analyzer.save_results_to_csv(results, filename="piano_analysis_output.csv")
        analyzer.save_results_to_txt(results, filename="piano_analysis_output.txt")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()