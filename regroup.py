import io
import re
from collections import defaultdict

# The raw text data provided by the user, now with a header.
# In a real-world scenario, you would read this from a file.
DATA = """
Time (s)   Note     Hand       Finger          Confidence  
-----------------------------------------------------------
3.82       C3       Left       5 (Pinky)       0.52        
3.83       C4       Right      1 (Thumb)       0.12        
4.43       C4       Right      1 (Thumb)       0.23        
5.03       E3       Left       3 (Middle)      0.55        
5.05       G4       Right      4 (Ring)        0.81        
5.61       G4       Right      4 (Ring)        0.79        
6.18       A4       Right      5 (Pinky)       0.75        
6.20       F3       Left       2 (Index)       0.52        
6.75       A4       Right      5 (Pinky)       0.79        
7.30       G4       Right      4 (Ring)        0.95        
7.31       E3       Left       3 (Middle)      0.27        
8.50       F4       Right      4 (Ring)        0.71        
8.51       D3       Left       4 (Ring)        0.34        
9.11       F4       Right      4 (Ring)        0.55        
9.69       E4       Right      3 (Middle)      0.69        
9.70       C3       Left       5 (Pinky)       0.33        
10.26      E4       Right      3 (Middle)      0.64        
10.85      G3       Left       1 (Thumb)       0.60        
10.88      D4       Right      2 (Index)       0.29        
11.42      D4       Right      1 (Thumb)       0.41        
12.02      C3       Left       5 (Pinky)       0.16        
12.02      C4       Right      1 (Thumb)       0.75        
13.27      G4       Right      5 (Pinky)       0.64        
13.28      C3       Left       5 (Pinky)       0.26        
13.87      G4       Right      5 (Pinky)       0.64        
14.45      F3       Right      2 (Index)       0.30        
14.46      F4       Right      4 (Ring)        0.23        
15.05      F4       Right      4 (Ring)        0.26        
15.64      C3       Left       5 (Pinky)       0.13        
15.65      E4       Right      3 (Middle)      0.40        
16.22      E4       Right      3 (Middle)      0.43        
16.83      G3       Left       1 (Thumb)       0.45        
16.84      D4       Right      2 (Index)       0.45        
18.06      C3       Left       5 (Pinky)       0.30        
18.09      G4       Right      5 (Pinky)       0.34        
18.62      G4       Right      5 (Pinky)       0.51        
19.20      F3       Right      2 (Index)       0.30        
19.23      F4       Right      4 (Ring)        0.23        
19.79      F4       Right      4 (Ring)        0.13        
20.37      C3       Left       5 (Pinky)       0.30        
20.39      E4       Right      3 (Middle)      0.31        
20.97      E4       Right      3 (Middle)      0.34        
21.52      G3       Left       1 (Thumb)       0.43        
21.54      D4       Right      2 (Index)       0.27        
22.74      C3       Right      1 (Thumb)       0.30        
22.75      C4       Right      1 (Thumb)       0.41        
23.33      C4       Right      1 (Thumb)       0.51        
23.90      E3       Left       3 (Middle)      0.30        
23.93      G4       Right      5 (Pinky)       0.23        
24.50      G4       Right      5 (Pinky)       0.25        
25.08      F3       Left       2 (Index)       0.30        
25.11      A4       Right      5 (Pinky)       0.57        
25.67      A4       Right      5 (Pinky)       0.43        
26.27      C3       Left       5 (Pinky)       0.30        
26.29      G4       Right      4 (Ring)        0.19        
27.49      D3       Left       4 (Ring)        0.30        
27.53      F4       Right      3 (Middle)      0.38        
28.12      F4       Right      3 (Middle)      0.40        
28.71      C3       Left       5 (Pinky)       0.30        
28.71      E4       Right      2 (Index)       0.92        
29.35      E4       Right      2 (Index)       0.90        
29.97      G3       Left       1 (Thumb)       0.15        
29.98      D4       Right      1 (Thumb)       0.19        
30.61      D4       Right      2 (Index)       0.30        
31.22      C3       Left       5 (Pinky)       0.30        
31.24      C4       Right      1 (Thumb)       0.30   
"""

def format_midi_summary(data_string, output_filename="midi_summary.txt", source_file="your_data.txt", segment_pause_threshold=1.2):
    """
    Parses performance data with a header and saves a detailed MIDI 
    transcription summary to a text file.
    """
    string_file = io.StringIO(data_string)
    
    # --- 1. Parse Header Metrics and find the start of note data ---
    performance_metrics = {}
    note_data_started = False
    
    all_notes = []
    
    for line in string_file:
        line = line.strip()
        if not line:
            continue

        if "-----------------" in line:
            note_data_started = True
            continue # Skip the separator line itself

        if not note_data_started:
            if ":" in line:
                # This is a metric line, e.g., "• BPM (Beats Per Minute): 91.1"
                key, value = line.split(':', 1)
                # Clean up the key by removing bullets and extra whitespace
                key = re.sub(r'^\s*•\s*', '', key).strip()
                performance_metrics[key] = value.strip()
        else:
            # We are in the note data section
            parts = line.split()
            try:
                # Combine finger parts, e.g., "5" and "(Pinky)"
                finger_info = f"{parts[3]} {parts[4]}"
                all_notes.append({
                    "time": float(parts[0]),
                    "note": parts[1],
                    "hand": parts[2],
                    "finger": finger_info,
                    "duration": 0  # Will be calculated next
                })
            except (ValueError, IndexError):
                continue # Skip any malformed note lines

    if not all_notes:
        print("No detailed note data found.")
        return

    # --- 2. Calculate duration for each note ---
    left_notes = sorted([n for n in all_notes if n['hand'] == 'Left'], key=lambda x: x['time'])
    right_notes = sorted([n for n in all_notes if n['hand'] == 'Right'], key=lambda x: x['time'])

    for hand_notes in [left_notes, right_notes]:
        for i in range(len(hand_notes) - 1):
            hand_notes[i]['duration'] = round(hand_notes[i+1]['time'] - hand_notes[i]['time'], 2)
        if hand_notes:
            hand_notes[-1]['duration'] = 0.25 

    # --- 3. Open output file and write Header ---
    with open(output_filename, 'w') as f:
        bpm = performance_metrics.get('BPM (Beats Per Minute)', 'N/A')
        total_notes = performance_metrics.get('Total Notes', len(all_notes))

        print("MIDI Transcription Summary", file=f)
        print(f"Source File: {source_file}", file=f)
        print(f"Total Notes: {total_notes}", file=f)
        print(f"Estimated Tempo: {bpm} BPM", file=f)
        print("========================================", file=f)
        print("\nNote Transcription (Split by Assumed Hand)\n", file=f)

        # --- 4. Group by timestamp and segment the output ---
        notes_by_time = defaultdict(dict)
        for note in all_notes:
            notes_by_time[note['time']][note['hand']] = note

        sorted_times = sorted(notes_by_time.keys())
        
        segment_start_time = sorted_times[0]
        last_time = sorted_times[0]

        def print_segment_header(start, end, file_handle):
            print(f"--- Segment ({start:.2f}s - {end:.2f}s) ---", file=file_handle)
            print(f"{'Time':<8}| {'Left Hand':<55}| {'Right Hand'}", file=file_handle)
            print(f"{'-'*8}|{'-'*55}|{'-'*55}", file=file_handle)

        for i, time in enumerate(sorted_times):
            if time - last_time > segment_pause_threshold and i > 0:
                print_segment_header(segment_start_time, last_time, f)
                
                segment_times = [t for t in sorted_times if segment_start_time <= t <= last_time]
                for seg_time in segment_times:
                    notes = notes_by_time[seg_time]
                    left_note = notes.get('Left')
                    right_note = notes.get('Right')
                    
                    left_str = f"{left_note['note']} (F:{left_note['finger']}, L:{left_note['duration']:.2f}s)" if left_note else ""
                    right_str = f"{right_note['note']} (F:{right_note['finger']}, L:{right_note['duration']:.2f}s)" if right_note else ""
                    
                    print(f"{seg_time:<7.2f}s| {left_str:<55}| {right_str}", file=f)
                print("\n", file=f)
                segment_start_time = time

            last_time = time
        
        # --- 5. Print the final (or only) segment ---
        print_segment_header(segment_start_time, last_time, f)
        segment_times = [t for t in sorted_times if segment_start_time <= t <= last_time]
        for seg_time in segment_times:
            notes = notes_by_time[seg_time]
            left_note = notes.get('Left')
            right_note = notes.get('Right')
            
            left_str = f"{left_note['note']} (F:{left_note['finger']}, L:{left_note['duration']:.2f}s)" if left_note else ""
            right_str = f"{right_note['note']} (F:{right_note['finger']}, L:{right_note['duration']:.2f}s)" if right_note else ""
            
            print(f"{seg_time:<7.2f}s| {left_str:<55}| {right_str}", file=f)
    
    print(f"Output successfully saved to {output_filename}")


if __name__ == "__main__":
    format_midi_summary(DATA, output_filename="midi_summary.txt")