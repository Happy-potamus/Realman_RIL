#!/usr/bin/env python3
import pretty_midi
import librosa
import numpy as np
from collections import Counter
import sys
import os

# --- Configuration ---
MIN_PATTERN_LENGTH = 8
MIN_PATTERN_OCCURRENCE = 2
# This is the most important setting to tune. A4 (69) is a good split point
# for separating mid-range chords. Try values between 60 (C4) and 72 (C5).
SPLIT_POINT = 69
# The pitch gap (in semitones) to identify as a hand separation. 12 = one octave.
HAND_SEPARATION_THRESHOLD = 12


def analyze_midi(midi_path: str) -> str | None:
    """
    Main function to load, analyze, and format a MIDI file.
    Returns the analysis as a string.
    """
    if not os.path.exists(midi_path):
        return f"Error: File not found at '{midi_path}'"

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        all_notes = [note for instrument in midi_data.instruments for note in instrument.notes]
        all_notes.sort(key=lambda x: x.start)
    except Exception as e:
        return f"Error loading MIDI file: {e}"

    if not all_notes:
        return "No notes found in this MIDI file."

    patterns = find_repeated_patterns(all_notes)
    segments = segment_song_structure(midi_data)
    output = format_analysis_output(
        midi_path, midi_data, all_notes, patterns, segments
    )
    return output

def find_repeated_patterns(notes: list) -> dict:
    """
    Identifies repeating sequences of notes (patterns/motifs).
    """
    pitches = [note.pitch for note in notes]
    sequences = Counter()
    for length in range(MIN_PATTERN_LENGTH, 13):
        for i in range(len(pitches) - length):
            sequence = tuple(pitches[i:i + length])
            sequences[sequence] += 1
    frequent_patterns = {
        seq: count
        for seq, count in sequences.items()
        if count >= MIN_PATTERN_OCCURRENCE
    }
    return sorted(frequent_patterns.items(), key=lambda item: item[1], reverse=True)

def segment_song_structure(midi_data: pretty_midi.PrettyMIDI) -> list:
    """
    Uses Librosa to find structural boundaries in the song.
    """
    try:
        waveform = midi_data.synthesize()
        sr = 44100
        chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
        R = librosa.segment.recurrence_matrix(chroma, mode="affinity", self=True)
        k_segments = min(10, R.shape[0] - 1)
        if k_segments > 1:
            boundaries = librosa.segment.agglomerative(R, k=k_segments)
            boundary_times = librosa.frames_to_time(boundaries, sr=sr)
        else:
            boundary_times = []
        segment_times = []
        start_time = 0.0
        for t in sorted(boundary_times):
            segment_times.append((start_time, t))
            start_time = t
        segment_times.append((start_time, midi_data.get_end_time()))
        return segment_times
    except Exception as e:
        print(f"Warning: Could not perform segmentation: {e}. Skipping segmentation.")
        return [(0.0, midi_data.get_end_time())]

def separate_hands_by_pitch_gap(notes_at_time: list, split_point: int, gap_threshold: int = HAND_SEPARATION_THRESHOLD) -> tuple[list, list]:
    """
    Separates notes into left/right hands using a pitch-gap and average pitch heuristic.
    """
    num_notes = len(notes_at_time)
    if num_notes == 0:
        return [], []
    if num_notes == 1:
        note = notes_at_time[0]
        return ([note], []) if note.pitch < split_point else ([], [note])

    sorted_notes = sorted(notes_at_time, key=lambda n: n.pitch)
    gaps = [sorted_notes[i+1].pitch - sorted_notes[i].pitch for i in range(num_notes - 1)]

    if gaps and max(gaps) >= gap_threshold:
        # If a large gap is found, split the hands there
        max_gap = max(gaps)
        split_index = gaps.index(max_gap) + 1
        lh_notes = sorted_notes[:split_index]
        rh_notes = sorted_notes[split_index:]
        return lh_notes, rh_notes
    else:
        # If notes are clustered, treat as a single chord and assign based on the chord's average pitch.
        average_pitch = sum(n.pitch for n in sorted_notes) / num_notes
        if average_pitch < split_point:
            return sorted_notes, []  # All notes assigned to left hand
        else:
            return [], sorted_notes  # All notes assigned to right hand

def format_analysis_output(midi_path, midi_data, notes, patterns, segments) -> str:
    """
    Assembles the final, human-readable text output from all analysis steps.
    """
    output_lines = []
    
    # --- High-Level Summary ---
    estimated_tempo = midi_data.estimate_tempo()
    output_lines.append("MIDI Transcription Summary")
    output_lines.append(f"Source File: {os.path.basename(midi_path)}")
    output_lines.append(f"Total Notes: {len(notes)}")
    output_lines.append(f"Estimated Tempo: {int(estimated_tempo)} BPM")
    output_lines.append("=" * 40)
    output_lines.append("\n--- Detected Melodic Patterns ---")

    # --- Patterns List ---
    pattern_map = {}
    for i, (pattern_pitches, count) in enumerate(patterns[:15], 1):
        note_names = " -> ".join([pretty_midi.note_number_to_name(p) for p in pattern_pitches])
        output_lines.append(f"Pattern {i} (occurs {count} times): {note_names}")
        pattern_map[pattern_pitches] = i

    # --- Prepare for Transcription ---
    note_start_to_pattern = {}
    pitches_tuple = tuple(n.pitch for n in notes)
    for i in range(len(pitches_tuple)):
        for pattern_pitches, pattern_id in pattern_map.items():
            if pitches_tuple[i:i + len(pattern_pitches)] == pattern_pitches:
                if notes[i].start not in note_start_to_pattern:
                    note_start_to_pattern[notes[i].start] = pattern_id

    # --- Tabular Transcription Logic ---
    output_lines.append("\n" + "=" * 40)
    output_lines.append("\nNote Transcription (Split by Assumed Hand)\n")

    for i, (start_t, end_t) in enumerate(segments, 1):
        segment_notes = [n for n in notes if start_t <= n.start < end_t]
        if not segment_notes:
            continue
            
        output_lines.append(f"--- Segment {i} ({start_t:.2f}s - {end_t:.2f}s) ---")
        header = f"{'Time':<8}| {'Left Hand':<55}| {'Right Hand'}"
        output_lines.append(header)
        output_lines.append(f"{'-'*8}|{'-'*55}|{'-'*55}")

        unique_times = sorted(list(set(n.start for n in segment_notes)))

        for time in unique_times:
            notes_at_time = [n for n in segment_notes if n.start == time]
            
            left_hand_notes, right_hand_notes = separate_hands_by_pitch_gap(notes_at_time, SPLIT_POINT)

            def format_notes(note_list):
                return ", ".join([f"{pretty_midi.note_number_to_name(n.pitch)} (V:{n.velocity}, L:{n.end - n.start:.2f}s)" for n in note_list])

            lh_str = format_notes(left_hand_notes)
            rh_str = format_notes(right_hand_notes)

            if time in note_start_to_pattern:
                annotation = f" <- Starts Pattern {note_start_to_pattern[time]}"
                if right_hand_notes:
                     rh_str += annotation
                else:
                     lh_str += annotation
            
            time_str = f"{time:.2f}s"
            output_lines.append(f"{time_str:<8}| {lh_str:<55}| {rh_str}")
        
        output_lines.append("")

    return "\n".join(output_lines)

def format_for_robot_task_plan(notes: list) -> str:
    """
    Formats the note sequence into a robot task plan, including duration.
    """
    output_lines = []
    header = f"{'Time (s)':<10}{'Robot Action'}"
    output_lines.append(header)
    
    notes.sort(key=lambda x: (x.start, x.pitch))
    
    for note in notes:
        time_str = f"{note.start:<10.2f}"
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        # Include duration "D" in the action string
        action_str = f"Play {note_name} (V:{note.velocity}, D:{duration:.2f}s)"
        output_lines.append(f"{time_str}{action_str}")
        
    return "\n".join(output_lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python midi_analyzer.py <path_to_your_midi_file.mid>")
    else:
        midi_file_path = sys.argv[1]
        base_name = os.path.splitext(os.path.basename(midi_file_path))[0]
        
        analysis_output = analyze_midi(midi_file_path)
        if analysis_output:
            print(analysis_output)
            output_path = f"{base_name}_analysis.txt"
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(analysis_output)
                print("\n" + "="*40)
                print(f"✅ Detailed analysis successfully saved to: {output_path}")
                print("="*40)
            except Exception as e:
                print(f"\nError: Could not save the analysis file. {e}")

        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            all_notes = [note for instrument in midi_data.instruments for note in instrument.notes]

            if all_notes:
                robot_plan_output = format_for_robot_task_plan(all_notes)
                robot_plan_path = f"{base_name}_robot_plan.txt"
                with open(robot_plan_path, "w", encoding="utf-8") as f:
                    f.write(robot_plan_output)
                print(f"✅ Robot task plan successfully saved to: {robot_plan_path}")
                print("="*40)
        except Exception:
            print(f"\nCould not generate the robot task plan. Please check the MIDI file.")