import os
import time
import json
from openai import OpenAI
from environment import BioCircuitEnv
from llm_judge import llm_judge, calculate_final_score
from dotenv import load_dotenv
# ... existing imports
load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")


import re
import json

def ask_agent(state):
    placed = state['circuit']
    
    # --- BUG FIX 1: SURGICAL FILTER ---
    # If any promoter is already in the circuit, remove all promoters from remaining list
    has_promoter = any("promoter" in p for p in placed)
    remaining = [
        p for p in state['available_parts'] 
        if p not in placed and not (has_promoter and "promoter" in p)
    ]
    
    # --- BUG FIX 2: DEFINE VARIABLES BEFORE USE ---
    system_msg = "You are a Synthetic Biology Expert. You ONLY respond in valid JSON."
    
    user_msg = f"""TASK: {state['task']}
TASK HINT: {state.get('hint', 'Build a functional transcription unit.')}
TARGET FLUORESCENCE: {state['target']}
CURRENT CIRCUIT (5' to 3'): {placed}
REMAINING ALLOWED PARTS: {remaining}

RULES:
1. Always start with a PROMOTER if the circuit is empty.
2. Place parts in biological order: Promoter first, Terminator last.
3. Choose only from REMAINING ALLOWED PARTS — do not invent parts.
4. Use the TASK HINT to decide which regulatory parts to include.

YOUR RESPONSE (valid JSON only, no extra text):
{{
  "action": {{"type": "place", "part": "exact_part_name_from_remaining_list"}},
  "reasoning": "Biological explanation referencing the specific molecular mechanism for this task."
}}"""

    # Now the variables exist when we call the API
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.0
    )
    
    raw = response.choices[0].message.content.strip()
    
    # --- BUG FIX 3: ROBUST PARSING ---
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            clean_json = re.sub(r'[\x00-\x1F\x7F]', ' ', match.group(0))
            return json.loads(clean_json)
        return {"action": {"type": "submit", "part": None}, "reasoning": "No JSON block"}
    except Exception as e:
        return {"action": {"type": "submit", "part": None}, "reasoning": f"Parser Error: {e}"}

def run_hackathon_eval(task_idx):
    env = BioCircuitEnv()
    state = env.reset(task_idx)
    print(f"\n--- Starting {state['task']} ---")
    
    agent_reasoning = []
    
    for step in range(10):
        decision = ask_agent(state)
        action = decision["action"]
        agent_reasoning.append(decision["reasoning"])
        
        state, reward, done = env.step(action)
        time.sleep(1.2)
        print(f"Step {step+1}: Placed {action.get('part')} | Reward: {reward}")
        
        if done:
            break

    # FINAL STEP: The LLM Judge evaluates the science
    print("\n[Professor Llama 3 is evaluating...]")
    verdict = llm_judge(
        circuit_parts=state['circuit'],
        mechanism_trace=agent_reasoning,
        fluorescence_out=state.get('fluorescence', 0.0), # Updated in environment
        math_reward=reward,
        task_id=task_idx + 1,
        task_name=state['task']
    )
    
    final_score = calculate_final_score(reward, verdict)
    print(f"Science Grade: {verdict.science_grade}")
    print(f"Critique: {verdict.critique}")
    print(f"FINAL SCORE (R x G): {final_score}")
    return final_score

if __name__ == "__main__":
    results = []
    for task_idx in range(15):
        score = run_hackathon_eval(task_idx)
        results.append({"task": task_idx + 1, "final_score": score})
    print("\n=== ALL 15 TASK RESULTS ===")
    for r in results:
        print(f"  Task {r['task']:02d}: Final Score = {r['final_score']:.4f}")