import json
import os
from reporter_logic import calculate_reporter_logic, DNAPart, PartType
from dense_rewards import check_dense_rewards

class BioCircuitEnv:
    def __init__(self):
        tasks_path = os.path.join(os.path.dirname(__file__), "tasks.json")
        with open(tasks_path, "r") as f:
            self.tasks = json.load(f)["tasks"]
        self.current_task_idx = 0
        self.task = self.tasks[0]
        self.circuit = []
        self.steps = 0
        self.done = False
        self.last_result = None

    def reset(self, task_index=0):
        self.current_task_idx = task_index
        self.task = self.tasks[task_index]
        self.circuit = [] 
        self.steps = 0
        self.done = False
        self.last_result = None
        return self.state()

    def state(self):
        return {
            "task": self.task["name"],
            "target": self.task["target_output"],
            "circuit": [p.part_type.value for p in self.circuit],
            "steps": self.steps,
            "available_parts": self.task["available_parts"],
            "fluorescence": self.last_result.fluorescence_output if self.last_result else 0.0,
            "math_reward": self.last_result.math_reward if self.last_result else 0.0,
            "hint": self.task.get("hint", "")
        }

    def step(self, action):
        self.steps += 1
        is_lethal = False
        step_reward = 0
        
        # 1. Determine if we are finishing
        is_submit = action.get("type") == "submit"
        is_max_steps = self.steps >= 10
        
        # 2. Process Part Placement (if not a pure submit)
        if not is_submit:
            part_name = action.get("part")
            if part_name:
                # Create and add the part
                new_part = DNAPart(part_type=PartType(part_name), slot_index=self.steps-1)
                
                # Check for Lethal Errors (Structural rules like Promoter location)
                checkpoint = check_dense_rewards(new_part, self.circuit, self.steps-1, self.current_task_idx + 1)
                
                self.circuit.append(new_part)
                step_reward = checkpoint.step_reward
                is_lethal = checkpoint.lethal_error
                
                # AUTO-DONE: If a terminator is placed, the transcription unit is complete
                if part_name == "terminator" or checkpoint.terminated:
                    self.done = True
        else:
            self.done = True

        # Force done if max steps reached
        if is_max_steps:
            self.done = True

        # 3. Final Evaluation Logic
        if self.done:
            result = calculate_reporter_logic(
                dna_sequence        = self.circuit,
                target_fluorescence = self.task["target_output"],
                target_part_count   = len(self.task["available_parts"])
            )
            self.last_result = result
            
            # Final Reward (R)
            reward = result.math_reward if not is_lethal else -10.0

            if self.done and not is_lethal:
                if self.current_task_idx < len(self.tasks) - 1:
                    self.current_task_idx += 1
        else:
            reward = step_reward

        return self.state(), reward, self.done