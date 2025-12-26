# Prompt: Generating a Hardcoded, Imperative Robot Script with a Python API and E-Stop
You are a senior robotics engineer writing a high-reliability execution script for a robot pianist. Your goal is to convert a structured **[Task Plan]** into a readable, modular, and completely hardcoded Python script that correctly uses a direct Python API and includes an explicit emergency stop feature.

The final script must **NOT** contain any data structures representing the plan. Instead, it must be a direct, imperative sequence of function calls.

## Input:
1.  **[Task Plan]:** A detailed, step-by-step plan for the song, outlining the timestamp, note, and duration for each action.
2.  **[Robot Primitives API]:** The script will be generated to use a pre-written library called `api_primitives`. The `player` object will have the following methods (primitives) available. All motion commands are blocking.
      * `player.connect()`: Establishes a connection to both robot arms. Returns boolean.
      * `player.prepare_arms()`: Powers on and enables joints for both arms. Returns boolean.
      * `player.move_to_home_position(hand='both')`: Moves arms to predefined home position. Returns boolean.
      * `player.calibrate_piano_base_position(hand='both', wait_for_input=False)`: Initializes the robot's position. If wait_for_input=False, assumes arms are at home position. Returns boolean.
      * `player.play_key(note: str, duration: float, prefer_hand=None)`: Plays a single key for a specified duration, automatically selecting the best hand unless prefer_hand is specified. Returns boolean.
      * `player.set_speeds(finger_speed=None, wrist_speed=None)`: Updates motion speeds. finger_speed range: 50-500, wrist_speed range: 10-100.
      * `player.return_to_base(hand='both')`: Moves the robot to a safe position after a normal performance.
      * `player.emergency_stop()`: A dedicated safe-shutdown call. The library automatically calls this on `Ctrl+C`.
      * `player.disconnect()`: Closes connections and cleans up resources.

## Instructions:
Your task is to generate a complete Python script by following these exact rules:

1.  **Imports:** At the top of the script, import the necessary libraries: `time` and the robot API: `from robot_pianist_api_library import RobotPianoPlayer`.

2.  **No Data Structures:** Do not copy the task plan data into the script. The plan is your source for writing the code, not for parsing at runtime.

3.  **Modular Functions:** Create a separate, dedicated function for each phase of the song (e.g., `play_intro(player)`, `play_verse_1(player)`, `play_chorus(player)`). Each function receives the `player` object as a parameter.

4.  **Imperative Code Generation:** Inside each function, write a direct sequence of `player` commands. Manage timing (`time.sleep()`) explicitly to match the task plan's timestamps. Use comments to mark sections or significant moments.

5.  **Optional Speed Configuration:** If the task plan specifies different speeds for different sections, call `player.set_speeds()` at the beginning of the relevant function.

6.  **Python Script Lifecycle and Main Orchestrator:** The script must be a standalone executable.
      * Define a `main()` function.
      * Inside `main()`, instantiate the robot object with optional speed parameters: `player = RobotPianoPlayer(finger_speed=200, wrist_speed=100)`.
      * Use a `try...finally` block to ensure a clean shutdown. The `KeyboardInterrupt` for emergency stop is handled automatically by a signal handler inside the library.
          * **`try` block:** 
            - Call `player.connect()` and check if successful
            - Call `player.prepare_arms()` and check if successful
            - Call `player.move_to_home_position(hand='both')` and check if successful
            - Call `player.calibrate_piano_base_position(hand='both', wait_for_input=False)` (assumes arms are already at home)
            - Optionally wait for the start time if needed
            - Call each phase function sequentially (e.g., `play_intro(player)`, `play_verse_1(player)`)
            - Call `player.return_to_base(hand='both')` on normal completion
          * **`finally` block:** This block must ensure a clean shutdown regardless of how the `try` block exits. Call `player.disconnect()`.
      * Use the standard `if __name__ == '__main__':` block to call your `main()` function.

7.  **Error Handling:** Check return values from `connect()`, `prepare_arms()`, `move_to_home_position()`, and `calibrate_piano_base_position()`. Return early from `main()` if any fail.

8.  **Comments:** Add clear comments to explain each section and phase of the performance.

## Example Structure:
```python
#!/usr/bin/env python3
import time
from robot_pianist_api_library import RobotPianoPlayer

def play_intro(player):
    """Play the introduction section."""
    # First phrase
    player.play_key('C4', 0.5)
    player.play_key('E4', 0.5)
    player.play_key('G4', 1.0)
    time.sleep(0.2)  # Brief pause
    
    # Second phrase
    player.play_key('F4', 0.5)
    player.play_key('A4', 0.5)
    player.play_key('C5', 1.0)

def play_verse_1(player):
    """Play verse 1 with slower tempo."""
    player.set_speeds(finger_speed=150, wrist_speed=80)
    
    player.play_key('D4', 0.8)
    player.play_key('E4', 0.8)
    # ... more notes

def main():
    player = RobotPianoPlayer(finger_speed=200, wrist_speed=100)
    
    try:
        # Initialize robot
        if not player.connect():
            print("Failed to connect. Exiting.")
            return
        if not player.prepare_arms():
            print("Failed to prepare arms. Exiting.")
            return
        if not player.move_to_home_position(hand='both'):
            print("Failed to move to home position. Exiting.")
            return
        if not player.calibrate_piano_base_position(hand='both', wait_for_input=False):
            print("Failed to calibrate. Exiting.")
            return
        
        print("Robot ready. Starting performance...")
        time.sleep(1)
        
        # Execute performance
        play_intro(player)
        play_verse_1(player)
        # ... more sections
        
        # Normal completion
        print("Performance complete!")
        player.return_to_base(hand='both')
        
    finally:
        player.disconnect()
        print("Script terminated.")

if __name__ == '__main__':
    main()
