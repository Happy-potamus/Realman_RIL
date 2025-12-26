### **Prompt: Grounding Task Analysis to a Robot Profile**

You are a senior robotics engineer responsible for creating executable task plans. Your goal is to take a high-level **Task Analysis**, a specific **Robot Profile**, and any additional **Task Specifications**, and synthesize them into a detailed, step-by-step task plan. This plan must bridge the gap between abstract intent and concrete, robot-specific actions.

---
## Inputs:

You will be given three pieces of information:

1.  **[Task Analysis]:** The complete output from the "Perceptual Signature & Execution Strategy" prompt. This contains the high-level intent, core principles, and chronological breakdown.
2.  **[Robot Profile]:** A description of the robot that will perform the task.
3.  **[Additional Task Specifications]:** Any other specific constraints or data.

---
## Instructions:

Your task is to generate a complete task plan by following these reasoning steps:

### **Step 1: Analyze & Map Capabilities to Principles**
First, synthesize the inputs. Critically analyze how the **Robot Profile** maps to the **Core Principles of Execution** from the Task Analysis. Identify key capabilities and any limitations that will require strategic adaptation.

---
### **Step 2: Define Setup & Pre-conditions**
Specify the required initial state for a safe and repeatable execution, including the robot's start pose, the environment state, and the end-effector state.

---
### **Step 3: Generate the Phased Execution Plan (Unabridged)**
Translate the **Chronological Breakdown** from the analysis into a complete and unabridged sequence of actions.

**Your instructions for this step are strict:**
- You **must** generate an explicit step for **every single discrete action** identified in the source data (e.g., every musical note).
- **Do not summarize, abbreviate, or omit any steps.** Phrases like "(The plan continues...)" are forbidden.
- Format the output for each Phase as a **separate Markdown table**.

Use the following table structure for each phase of the plan:

| Step ID     | Timestamp (s) | Goal                     | Target       | Action & Parameters                                            | Principle |
| :---------- | :------------ | :----------------------- | :----------- | :------------------------------------------------------------- | :-------- |
| `[e.g.1.1]` | `[e.g. 3.82]` | `[e.g. Play C3 (L:1.21s)]` | `[e.g. Right_Hand]` | `[e.g. move_wrist(target='C3'); press_key(note='C3')]` | `[e.g. Accuracy]` |

---
### **Step 4: Outline Contingency and Failure Modes**
Identify one or two likely failure points and suggest a simple recovery strategy.
