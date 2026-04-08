import os
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
    remaining = [
        p for p in state['available_parts']
        if p not in placed and not (any("promoter" in x for x in placed) and "promoter" in p)
    ]

    # HARD GUARD: Step 1 must ALWAYS be a promoter
    if not placed:
        promoter_options = [p for p in remaining if "promoter" in p]
        if promoter_options:
            return {
                "action": {"type": "place", "part": promoter_options[0]},
                "reasoning": "Promoter must always be placed first as the RNA Polymerase initiation site."
            }
    
    # --- BUG FIX 2: DEFINE VARIABLES BEFORE USE ---
    system_msg = (
        "You are a Synthetic Biology Expert designing genetic circuits. "
        "You ONLY respond in valid JSON. "
        "ABSOLUTE RULE 1: If CURRENT CIRCUIT is empty [], your response "
        "MUST place a promoter. The very first part MUST always be a promoter. "
        "This is non-negotiable — RNA Polymerase cannot start without a promoter. "
        "ABSOLUTE RULE 2: Only choose parts from REMAINING ALLOWED PARTS. "
        "ABSOLUTE RULE 3: Never place a terminator unless a reporter_gene or "
        "structural_gene is already in CURRENT CIRCUIT. "
        "TASK LOGIC GUIDE: "
        "- CAP/cAMP tasks: Promoter → cap_binding_site → reporter_gene → terminator. "
        "- Enhancer tasks: Promoter → enhancer → reporter_gene → terminator. "
        "- Repression tasks: Promoter → operator → repressor_gene → reporter_gene → terminator. "
        "- Basic tasks: Promoter → reporter_gene → terminator. "
        "Use the TASK HINT to determine which type applies."
    )
    
    user_msg = f"""TASK: {state['task']}
CURRENT CIRCUIT: {placed}
REMAINING ALLOWED PARTS: {remaining}

Use the TASK HINT to construct your circuit logic. 
HINT: {state.get('hint', '')}

YOUR RESPONSE (JSON ONLY):
{{
  "action": {{"type": "place", "part": "part_name"}},
  "reasoning": "Specifically describe the biological mechanism."
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
            parsed = json.loads(clean_json)
            
            part = parsed.get("action", {}).get("part")
            # Enforce that the part must be in the available_parts or remaining
            if part and part not in state['available_parts']:
                if part == "repressor" and "repressor_gene" in remaining:
                    parsed["action"]["part"] = "repressor_gene"
                else:
                    # Fallback to the first available remaining part to prevent crashing
                    parsed["action"]["part"] = remaining[0] if remaining else None
                    
            return parsed
        return {"action": {"type": "submit", "part": None}, "reasoning": "No JSON block"}
    except Exception as e:
        return {"action": {"type": "submit", "part": None}, "reasoning": f"Parser Error: {e}"}

def run_hackathon_eval(task_idx):
    import time
    env = BioCircuitEnv()
    state = env.reset(task_idx)
    print(f"\n--- Starting {state['task']} ---")

    agent_reasoning = []

    for step in range(10):
        decision = ask_agent(state)
        action = decision["action"]
        agent_reasoning.append(decision.get("reasoning", ""))

        state, reward, done = env.step(action)
        print(f"Step {step+1}: Placed {action.get('part')} | Reward: {reward}")
        time.sleep(1.2)

        if done:
            break

    print("\n[Professor Llama 3 is evaluating...]")
    verdict = llm_judge(
        circuit_parts=state['circuit'],
        mechanism_trace=agent_reasoning,
        fluorescence_out=state.get('fluorescence', 0.0),
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
    all_results = []
    for task_idx in range(15):
        score = run_hackathon_eval(task_idx)
        all_results.append({"task": task_idx + 1, "score": score or 0.0})

    print("\n" + "="*52)
    print("    HACKATHON FINAL RESULTS — ALL 15 TASKS")
    print("="*52)
    for r in all_results:
        bar = "█" * int(r["score"] * 2) if r["score"] > 0 else ""
        print(f"  Task {r['task']:02d}: {r['score']:8.4f}  {bar}")
    avg = sum(r["score"] for r in all_results) / 15
    print(f"\n  AVERAGE SCORE : {avg:.4f}")
    print("="*52)