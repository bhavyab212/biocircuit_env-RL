import os
import re
import json
import time

from openai import OpenAI
from dotenv import load_dotenv
from environment import BioCircuitEnv
from llm_judge import llm_judge, calculate_final_score

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ── OpenEnv structured log helpers ──────────────────────────────────────────

def log_start(task, env_name, model):
    print(f"[START] task={task} env=SynBio-RL model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Agent ────────────────────────────────────────────────────────────────────

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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            clean_json = re.sub(r'[\x00-\x1F\x7F]', ' ', match.group(0))
            parsed = json.loads(clean_json)

            part = parsed.get("action", {}).get("part")
            # Enforce that the part must be in available_parts
            if part and part not in state['available_parts']:
                if part == "repressor" and "repressor_gene" in remaining:
                    parsed["action"]["part"] = "repressor_gene"
                else:
                    parsed["action"]["part"] = remaining[0] if remaining else None

            return parsed
        return {"action": {"type": "submit", "part": None}, "reasoning": "No JSON block"}
    except Exception as e:
        return {"action": {"type": "submit", "part": None}, "reasoning": f"Parser Error: {e}"}


# ── Evaluation loop ──────────────────────────────────────────────────────────

def run_hackathon_eval(task_idx):
    env = BioCircuitEnv()
    state = env.reset(task_idx)
    print(f"\n--- Starting {state['task']} ---")
    log_start(state['task'], 'SynBio-RL', MODEL_NAME)

    agent_reasoning = []
    rewards_list = []

    for step in range(10):
        decision = ask_agent(state)
        action = decision["action"]
        agent_reasoning.append(decision.get("reasoning", ""))

        state, reward, done = env.step(action)
        norm_reward = min(max(round(reward / 10.0, 4), 0.0), 1.0)
        log_step(step + 1, action.get('part', 'submit'), norm_reward, done)
        rewards_list.append(norm_reward)
        print(f"Step {step+1}: Placed {action.get('part')} | Reward: {reward}")
        time.sleep(0.5)

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

    norm_final = min(max(round(final_score / 10.0, 4), 0.0), 1.0)
    log_end(norm_final > 0.1, env.steps, norm_final, rewards_list)

    return norm_final


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []
    for task_idx in range(15):
        score = run_hackathon_eval(task_idx)
        all_results.append(score or 0.0)
        time.sleep(0.5)

    print("\n=== ALL 15 TASK RESULTS ===")
    for i, s in enumerate(all_results):
        print(f"  Task {i+1:02d}: Final Score = {s:.4f}")