# Constrained Imitation

LLM-powered code generation for robot piano playing using video demonstrations and constraint-based planning.

## Quick Start

**To run the pipeline, navigate to the `api` directory:**

```bash
cd api
```

**Then follow the instructions in [`api/README.md`](api/README.md)** for setup and execution details.

The main entry point is `api/run.py`, which orchestrates the three-phase pipeline:
1. **Task Planning**: Analyzes video and notes to create a task plan
2. **Constraint Planning**: Applies physical constraints and robot configuration
3. **Code Generation**: Generates executable robot script

## Project Structure

### Core API Libraries

- **`api_primitives.py`**: Dual-hand robot piano player library providing high-level primitives for controlling both robot arms. Uses direct Python API (non-ROS) with collision avoidance and piano keyboard mapping.

### Planning & Configuration Files

- **`api/physical_constraint.txt`** & **`physical-constraint-prompt.md`**: Physical constraints and prompts for grounding task analysis to robot capabilities. Defines limitations, safety requirements, and adaptation strategies for the robot.

- **`api/coding_module.txt`** & **`coding-module.md`**: Instructions and prompts for generating hardcoded, imperative robot scripts. Specifies the API structure, code generation rules, and best practices.

- **`api/task_prompt.txt`** & **`task-prompt.md`**: Task analysis prompts for intent-driven planning. Helps extract the "perceptual signature" and core principles from human demonstrations.

- **`api/robot_config.txt`**: Robot-specific configuration including capabilities, joint limits, workspace constraints, and calibration parameters.

### Processing Modules

- **`api/task_planning.py`**: Module for analyzing video demonstrations and text notes using Gemini API. Extracts task intent and creates high-level execution strategies.

- **`api/constraint_planning.py`**: Applies physical constraints and robot configuration to task plans, generating constraint-aware execution plans.

- **`api/generate_code.py`**: Converts constraint-aware plans into executable Python robot scripts using the primitives API.

- **`api/feedback.py`**: Analyzes robot performance videos and generates improved code versions based on feedback.

### MIDI & Music Processing

- **`midi.py`**: MIDI file processing utilities for parsing and analyzing musical data.

## Workflow

1. **Prepare inputs**: 
   - Video file (`.mp4`) showing the piano performance
   - MIDI file (`.mid`) from the piano recording

2. **Convert MIDI to text**: 
   ```bash
   python midi.py <your_file.mid>
   ```
   This generates a text analysis file (e.g., `your_file_analysis.txt`) containing note transcription, timing, and musical structure.

3. **Generate robot code**: 
   ```bash
   cd api
   python run.py
   ```
   The pipeline uses the video file and the generated text file to create an executable robot script (`play.py`). See `api/README.md` for configuration details.

4. **Execute generated code**: Run the generated `api/play.py` script to control the robot and perform the piece.

5. **Iterate with feedback**: Record a video of the robot's performance and use `api/feedback.py` to analyze it. This will generate an improved version of the code based on the robot's actual execution.

For detailed setup and usage instructions, see **[`api/README.md`](api/README.md)**.
