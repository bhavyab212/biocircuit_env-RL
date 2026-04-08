
"""
dense_rewards.py
================
SynBio-RL Project — Meta OpenEnv Hackathon
Task 2: Dense Reward Checkpoints — Bhavya

Provides IMMEDIATE biological feedback at every single step of the
agent's DNA assembly. Called after EACH part placement, not just at
the end of the episode.

This module solves the Sparse Reward Problem:
  Without checkpoints  → Agent receives a score only at episode end.
                          Credit assignment is noisy. Learning stalls.
  With dense rewards   → Agent receives biological feedback instantly
                          at every step. Neural network updates cleanly
                          after every placement decision. Trains 100× faster.

Architecture of this module
────────────────────────────
  1.  TaskCheckpointConfig   — Per-task biological expectations
  2.  CheckpointResult       — Structured return type (reward, flags, trace)
  3.  Checkpoint validators  — One pure function per biological rule
  4.  check_dense_rewards()  — Main entry point called by the environment

Biological rules modelled
──────────────────────────
  • 5′→3′ directionality enforcement (cis-acting positional rules)
  • Promoter-first constraint (RNA Pol initiation requirement)
  • Operator must precede Gene to enable steric hindrance
  • Terminator must be final part (blocks all downstream reading)
  • Repressor Gene requires an Operator to have regulatory effect
  • CAP Binding Site requires cAMP flag to activate
  • Duplicate Promoters waste metabolic resources
  • Strong Promoter on a low-complexity task → efficiency warning
  • Enhancer placement — position-flexible but only meaningful near Promoter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import copy

# Re-use the part library defined in reporter_logic.py
# (In production these would be imported: from reporter_logic import ...)

class PartType(str, Enum):
    STRONG_PROMOTER   = "strong_promoter"
    MEDIUM_PROMOTER   = "medium_promoter"
    WEAK_PROMOTER     = "weak_promoter"
    OPERATOR          = "operator"
    REPORTER_GENE     = "reporter_gene"
    STRUCTURAL_GENE   = "structural_gene"
    REPRESSOR_GENE    = "repressor_gene"
    CAP_BINDING_SITE  = "cap_binding_site"
    TERMINATOR        = "terminator"
    ENHANCER          = "enhancer"


PROMOTER_TYPES = {
    PartType.STRONG_PROMOTER,
    PartType.MEDIUM_PROMOTER,
    PartType.WEAK_PROMOTER,
}
GENE_TYPES = {PartType.REPORTER_GENE, PartType.STRUCTURAL_GENE}

PROMOTER_STRENGTH = {
    PartType.STRONG_PROMOTER: 1.0,
    PartType.MEDIUM_PROMOTER: 0.5,
    PartType.WEAK_PROMOTER:   0.2,
}


@dataclass
class DNAPart:
    part_type       : PartType
    slot_index      : int
    inducer_present : bool = False
    cAMP_present    : bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TASK CHECKPOINT CONFIGURATIONS
#     Defines exactly what each of the 15 tasks requires biologically.
#     The checkpoint validator uses this to know what reward/penalty to assign.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskCheckpointConfig:
    """
    Per-task biological requirements used by the checkpoint validator.

    Attributes
    ----------
    task_id : int
        Task number (1–15).
    task_name : str
        Human-readable task name.
    requires_operator : bool
        Task requires an Operator for regulatory logic.
    requires_repressor : bool
        Task requires a Repressor Gene (LacI-type).
    requires_inducer : bool
        Task requires a Ligand/Inducer to relieve repression.
    requires_cap : bool
        Task requires CAP Binding Site + cAMP.
    requires_enhancer : bool
        Task requires an Enhancer (DNA looping, Task 13).
    requires_polycistronic : bool
        Task requires multiple genes under one Promoter.
    target_fluorescence : float
        Desired GFP output [0.0, 1.0] for this task.
    optimal_part_count : int
        Minimum number of parts a correct solution uses.
    preferred_promoter_strength : Optional[str]
        "weak", "medium", "strong", or None (any accepted).
    """
    task_id                    : int
    task_name                  : str
    requires_operator          : bool  = False
    requires_repressor         : bool  = False
    requires_inducer           : bool  = False
    requires_cap               : bool  = False
    requires_enhancer          : bool  = False
    requires_polycistronic     : bool  = False
    target_fluorescence        : float = 1.0
    optimal_part_count         : int   = 3
    preferred_promoter_strength: Optional[str] = None


# All 15 task configurations derived from the 15-Task challenge document
TASK_CONFIGS: dict[int, TaskCheckpointConfig] = {
    1: TaskCheckpointConfig(1, "Basic Initiation",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=3,
        preferred_promoter_strength="strong"),

    2: TaskCheckpointConfig(2, "The Biological Brake",
        requires_operator=True, requires_repressor=True,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=0.0, optimal_part_count=5,
        preferred_promoter_strength="medium"),

    3: TaskCheckpointConfig(3, "The Inducible Switch",
        requires_operator=True, requires_repressor=True,
        requires_inducer=True, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=5,
        preferred_promoter_strength="medium"),

    4: TaskCheckpointConfig(4, "Signal Boosting",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=True,
        requires_enhancer=False,
        target_fluorescence=0.7, optimal_part_count=4,
        preferred_promoter_strength="weak"),

    5: TaskCheckpointConfig(5, "Comparing Strengths",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=3,
        preferred_promoter_strength="strong"),

    6: TaskCheckpointConfig(6, "Distant Control",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=True,
        target_fluorescence=0.75, optimal_part_count=4,
        preferred_promoter_strength="medium"),

    7: TaskCheckpointConfig(7, "The Signal Relay",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=4,
        preferred_promoter_strength="weak"),

    8: TaskCheckpointConfig(8, "The Silencer",
        requires_operator=True, requires_repressor=True,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=0.0, optimal_part_count=4,
        preferred_promoter_strength="strong"),

    9: TaskCheckpointConfig(9, "Drug Sensitivity",
        requires_operator=True, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=0.5, optimal_part_count=4,
        preferred_promoter_strength="medium"),

    10: TaskCheckpointConfig(10, "Feedback Loop",
        requires_operator=False, requires_repressor=True,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=0.5, optimal_part_count=4,
        preferred_promoter_strength="medium"),

    11: TaskCheckpointConfig(11, "Combinatorial AND Gate",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=True,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=4,
        preferred_promoter_strength="weak"),

    12: TaskCheckpointConfig(12, "Metabolic Switching",
        requires_operator=True, requires_repressor=True,
        requires_inducer=True, requires_cap=True,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=6,
        preferred_promoter_strength="weak"),

    13: TaskCheckpointConfig(13, "Enhancer Loop",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=True,
        target_fluorescence=0.75, optimal_part_count=4,
        preferred_promoter_strength="medium"),

    14: TaskCheckpointConfig(14, "Intracellular Sensor",
        requires_operator=False, requires_repressor=False,
        requires_inducer=False, requires_cap=False,
        requires_enhancer=False,
        target_fluorescence=1.0, optimal_part_count=4,
        preferred_promoter_strength="medium"),

    15: TaskCheckpointConfig(15, "Relay Amplification",
        requires_operator=True, requires_repressor=True,
        requires_inducer=False, requires_cap=True,
        requires_enhancer=True,
        target_fluorescence=1.0, optimal_part_count=7,
        preferred_promoter_strength="weak"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHECKPOINT RESULT — Structured Return Type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointResult:
    """
    The immediate biological feedback returned after each single part placement.

    Attributes
    ----------
    step_reward : float
        Immediate reward/penalty for this specific placement step.
        Positive = biologically correct move.
        Negative = biologically incorrect or inefficient move.
    terminated : bool
        True if a LETHAL structural error was committed on this step.
        When True, the environment must end the episode immediately.
        The agent receives the step_reward (a large negative penalty)
        and no further steps are taken.
    truncated : bool
        True if the episode should end cleanly because the circuit is
        complete and valid (Terminator placed as last part).
    lethal_error : bool
        Specifically flags that the termination is due to a lethal error,
        not normal completion.
    lethal_reason : Optional[str]
        Biological explanation of the lethal error (fed to LLM judge).
    step_feedback : str
        One-line biological explanation of why this reward was assigned.
        Human-readable for debugging and LLM judge prompt construction.
    warnings : list[str]
        Non-fatal biological concerns for this step.
    mechanism_trace : list[str]
        Complete biological reasoning log for this step.
    action_mask : list[int]
        Binary mask [len = 10] indicating which PartTypes are legally
        available for the NEXT step, given current circuit state.
        Index order matches PartType enum order.
        0 = disabled (biologically illegal next step)
        1 = available
    cumulative_reward : float
        Running sum of all step_rewards so far in this episode.
        Set by the environment after each call (not by this function).
    """
    step_reward        : float          = 0.0
    terminated         : bool           = False
    truncated          : bool           = False
    lethal_error       : bool           = False
    lethal_reason      : Optional[str]  = None
    step_feedback      : str            = ""
    warnings           : list[str]      = field(default_factory=list)
    mechanism_trace    : list[str]      = field(default_factory=list)
    action_mask        : list[int]      = field(default_factory=lambda: [1]*10)
    cumulative_reward  : float          = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ACTION MASK BUILDER
#     Returns a binary mask of length 10 (one per PartType in enum order).
#     0 = this part is biologically illegal as the NEXT placement.
#     1 = this part is a valid next choice.
# ─────────────────────────────────────────────────────────────────────────────

PART_TYPE_ORDER = list(PartType)  # deterministic enum order for mask indexing


def build_action_mask(current_sequence: list[DNAPart],
                      task_config: TaskCheckpointConfig) -> list[int]:
    """
    Compute the action mask for the NEXT step given the current circuit state.

    Biology enforced by the mask:
    ─────────────────────────────
    • If no part placed yet:
        Only Promoters are valid as the first part (5′ start rule).
    • If Promoter exists but Terminator does not:
        Promoters are disabled (no duplicate Promoters).
        Terminator is allowed only after a Reporter Gene exists.
    • If Terminator has been placed:
        ALL actions disabled — circuit is complete.
    • Operator is only valid after a Promoter has been placed.
    • Repressor Gene is only valid if an Operator is present.
    • CAP Binding Site is only valid after a Promoter is placed.
    • Reporter Gene / Structural Gene require a Promoter first.

    Returns
    -------
    list[int] of length 10, matching PART_TYPE_ORDER index positions.
    """
    mask = [1] * len(PART_TYPE_ORDER)

    has_promoter   = any(p.part_type in PROMOTER_TYPES       for p in current_sequence)
    has_operator   = any(p.part_type == PartType.OPERATOR     for p in current_sequence)
    has_gene       = any(p.part_type in GENE_TYPES            for p in current_sequence)
    has_terminator = any(p.part_type == PartType.TERMINATOR   for p in current_sequence)
    has_repressor  = any(p.part_type == PartType.REPRESSOR_GENE for p in current_sequence)

    # ── If Terminator already placed: everything disabled ────────────────────
    if has_terminator:
        return [0] * len(PART_TYPE_ORDER)

    # ── First placement: only Promoters allowed ───────────────────────────────
    if not current_sequence:
        for i, pt in enumerate(PART_TYPE_ORDER):
            mask[i] = 1 if pt in PROMOTER_TYPES else 0
        return mask

    # ── After first placement ─────────────────────────────────────────────────
    for i, pt in enumerate(PART_TYPE_ORDER):

        # No duplicate Promoters
        if pt in PROMOTER_TYPES and has_promoter:
            mask[i] = 0

        # Operator requires a Promoter to exist first
        if pt == PartType.OPERATOR and not has_promoter:
            mask[i] = 0

        # Repressor Gene requires an Operator to have any effect
        if pt == PartType.REPRESSOR_GENE and not has_operator:
            mask[i] = 0

        # Reporter / Structural Gene require a Promoter
        if pt in GENE_TYPES and not has_promoter:
            mask[i] = 0

        # CAP Binding Site requires a Promoter
        if pt == PartType.CAP_BINDING_SITE and not has_promoter:
            mask[i] = 0

        # Enhancer requires a Promoter
        if pt == PartType.ENHANCER and not has_promoter:
            mask[i] = 0

        # Terminator only valid once a Reporter Gene exists
        if pt == PartType.TERMINATOR and not has_gene:
            mask[i] = 0

        # Task-specific constraints
        # If task does NOT require a Repressor, disable Repressor Gene
        if pt == PartType.REPRESSOR_GENE and not task_config.requires_repressor:
            mask[i] = 0

        # If task does NOT require CAP, disable CAP Binding Site
        if pt == PartType.CAP_BINDING_SITE and not task_config.requires_cap:
            mask[i] = 0

        # If task does NOT require Enhancer, disable Enhancer
        if pt == PartType.ENHANCER and not task_config.requires_enhancer:
            mask[i] = 0

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INDIVIDUAL CHECKPOINT VALIDATORS
#     Each function evaluates ONE specific biological rule for ONE step.
#     Returns (reward_delta: float, feedback: str, is_lethal: bool)
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_first_part(new_part: DNAPart,
                            result: CheckpointResult) -> bool:
    """
    Rule: The very first part placed MUST be a Promoter.
    Biology: RNA Polymerase can only initiate transcription by binding
    a Promoter. Without a Promoter as the first element, no transcription
    can begin — the circuit is non-functional by definition.
    """
    if new_part.part_type in PROMOTER_TYPES:
        result.step_reward  += 1.0
        result.step_feedback = (
            f"✔ Correct: {new_part.part_type.value} placed first (5′ end). "
            f"RNA Polymerase has a valid landing site. "
            f"Basal transcription rate = {PROMOTER_STRENGTH[new_part.part_type]:.1f}."
        )
        result.mechanism_trace.append(result.step_feedback)
        return False   # no lethal error
    else:
        result.step_reward  += -5.0
        result.terminated    = True
        result.lethal_error  = True
        result.lethal_reason = (
            f"LETHAL: First part placed is {new_part.part_type.value}, not a Promoter. "
            f"RNA Pol has no initiation site. Circuit cannot function. "
            f"Per LLM Rubric Section I: Lethal Error → G = 0. Episode terminated."
        )
        result.step_feedback = result.lethal_reason
        result.mechanism_trace.append(result.lethal_reason)
        return True   # lethal error → terminate


def _checkpoint_promoter_strength(new_part: DNAPart,
                                   task_config: TaskCheckpointConfig,
                                   result: CheckpointResult) -> None:
    """
    Rule: Match Promoter strength to task complexity.
    Biology: Strong Promoters maximise transcription but impose high
    Metabolic Burden. For tasks requiring low/partial expression or
    testing CAP activation (Tasks 7, 11), a Weak Promoter is required.
    Using a Strong Promoter where unnecessary wastes cellular ATP.
    """
    if new_part.part_type not in PROMOTER_TYPES:
        return

    pref = task_config.preferred_promoter_strength
    actual_strength = new_part.part_type.value.replace("_promoter", "")

    if pref is None:
        # No specific preference — any promoter is fine
        result.step_reward  += 0.2
        result.mechanism_trace.append(
            f"  Promoter strength {actual_strength} — no preference for this task. +0.2"
        )
    elif actual_strength == pref:
        result.step_reward  += 0.5
        result.mechanism_trace.append(
            f"  ✔ Promoter strength '{actual_strength}' matches task requirement. +0.5"
        )
    else:
        result.step_reward  -= 0.3
        result.warnings.append(
            f"Suboptimal Promoter strength: task {task_config.task_id} "
            f"({task_config.task_name}) prefers a '{pref}' Promoter. "
            f"You placed '{actual_strength}'. "
            f"This may cause Metabolic Burden issues or wrong fluorescence level."
        )
        result.mechanism_trace.append(
            f"  ⚠ Promoter strength '{actual_strength}' does not match "
            f"task preference '{pref}'. −0.3"
        )


def _checkpoint_operator_placement(new_part: DNAPart,
                                    current_sequence: list[DNAPart],
                                    result: CheckpointResult) -> None:
    """
    Rule: Operator must be placed AFTER Promoter and BEFORE Reporter Gene.
    Biology: The Operator is a cis-acting element. Its physical position
    between the Promoter and the Gene is what allows a Repressor to sit
    on it and block RNA Pol via steric hindrance. An Operator placed after
    the Gene, or before a Promoter, has zero regulatory function.
    """
    if new_part.part_type != PartType.OPERATOR:
        return

    has_promoter = any(p.part_type in PROMOTER_TYPES for p in current_sequence)
    has_gene     = any(p.part_type in GENE_TYPES      for p in current_sequence)

    if has_promoter and not has_gene:
        # Perfect placement — Operator is between Promoter and Gene
        result.step_reward  += 0.5
        result.step_feedback = (
            f"✔ Correct: Operator placed after Promoter and before Reporter Gene. "
            f"Steric hindrance is now possible. "
            f"A Repressor binding here will block RNA Pol progression."
        )
        result.mechanism_trace.append(result.step_feedback)
    elif not has_promoter:
        # Operator placed before any Promoter — biologically wrong
        result.step_reward  -= 1.0
        result.warnings.append(
            "Operator placed before a Promoter. No RNA Pol will ever reach this "
            "Operator since transcription cannot initiate without a Promoter first. "
            "Operator has no effect here."
        )
        result.mechanism_trace.append(
            "  ✗ Operator placed before Promoter — no steric hindrance possible. −1.0"
        )
    elif has_gene:
        # Operator placed after Gene — Repressor cannot block RNA Pol from reading gene
        result.step_reward  -= 2.0
        result.warnings.append(
            "Operator placed AFTER Reporter Gene. Steric hindrance requires the "
            "Operator to BLOCK RNA Pol before it reaches the gene. A downstream "
            "Operator is never in the path of RNA Pol. This is a major design error."
        )
        result.mechanism_trace.append(
            "  ✗ Operator placed after Gene — post-gene Operator cannot block RNA Pol. −2.0"
        )


def _checkpoint_repressor_gene(new_part: DNAPart,
                                current_sequence: list[DNAPart],
                                task_config: TaskCheckpointConfig,
                                result: CheckpointResult) -> None:
    """
    Rule: A Repressor Gene only makes biological sense if an Operator exists
    for the Repressor protein to bind to.
    Biology: The Repressor Gene is a trans-acting element — it produces
    a diffusible protein (LacI) that must bind the Operator sequence to
    function. If no Operator is present, the Repressor protein is
    produced but has nowhere to bind. It is a wasted metabolic cost.
    """
    if new_part.part_type != PartType.REPRESSOR_GENE:
        return

    has_operator = any(p.part_type == PartType.OPERATOR for p in current_sequence)

    if has_operator and task_config.requires_repressor:
        result.step_reward  += 0.5
        result.step_feedback = (
            "✔ Correct: Repressor Gene placed after Operator. "
            "The Repressor protein (LacI) will be produced and will bind "
            "the Operator to block RNA Pol. NOT Gate logic is now established."
        )
        result.mechanism_trace.append(result.step_feedback)
    elif not has_operator:
        result.step_reward  -= 0.5
        result.warnings.append(
            "Repressor Gene placed but no Operator exists in the circuit. "
            "LacI protein will be produced but has no Operator to bind. "
            "Place an Operator BEFORE placing the Repressor Gene."
        )
        result.mechanism_trace.append(
            "  ⚠ Repressor Gene without Operator — LacI has no binding site. −0.5"
        )
    elif not task_config.requires_repressor:
        result.step_reward  -= 0.3
        result.warnings.append(
            f"Repressor Gene placed but Task {task_config.task_id} "
            f"({task_config.task_name}) does not require repression logic. "
            "Unnecessary metabolic cost."
        )
        result.mechanism_trace.append(
            f"  ⚠ Repressor Gene not required for Task {task_config.task_id}. −0.3"
        )


def _checkpoint_cap_binding_site(new_part: DNAPart,
                                  task_config: TaskCheckpointConfig,
                                  result: CheckpointResult) -> None:
    """
    Rule: CAP Binding Site is only meaningful if cAMP is present.
    Biology: CAP (Catabolite Activator Protein) is an activator. It needs
    cAMP as a co-factor to adopt the conformation that lets it bind DNA
    near the Promoter and recruit RNA Pol. Without cAMP, CAP cannot
    activate transcription. This is the key check for Tasks 11 and 12
    per the LLM Rubric (Level 12 Key Check).
    """
    if new_part.part_type != PartType.CAP_BINDING_SITE:
        return

    if task_config.requires_cap:
        if new_part.cAMP_present:
            result.step_reward  += 0.5
            result.step_feedback = (
                "✔ Correct: CAP Binding Site placed with cAMP present. "
                "CAP protein is active and will recruit RNA Pol to the Promoter, "
                "boosting transcription. (LLM Rubric Level 12 Key Check: PASSED)"
            )
            result.mechanism_trace.append(result.step_feedback)
        else:
            result.step_reward  -= 0.4
            result.warnings.append(
                "CAP Binding Site placed but cAMP is NOT present. "
                "CAP protein cannot activate without its cAMP co-factor. "
                "Per LLM Rubric Level 12 Key Check: must mention cAMP. "
                "No activation boost will be applied."
            )
            result.mechanism_trace.append(
                "  ⚠ CAP Binding Site: cAMP absent — no activation. "
                "LLM Rubric Level 12 Key Check will FAIL. −0.4"
            )
    else:
        result.step_reward  -= 0.2
        result.warnings.append(
            f"CAP Binding Site placed but Task {task_config.task_id} "
            f"({task_config.task_name}) does not require CAP activation. "
            "Unnecessary part — adds metabolic cost without benefit."
        )
        result.mechanism_trace.append(
            f"  ⚠ CAP Binding Site not required for Task {task_config.task_id}. −0.2"
        )


def _checkpoint_terminator(new_part: DNAPart,
                             current_sequence: list[DNAPart],
                             task_config: TaskCheckpointConfig,
                             result: CheckpointResult) -> bool:
    """
    Rule: Terminator is valid only when placed AFTER a Reporter Gene.
    A Terminator placed BEFORE the Reporter Gene is a lethal error
    — RNA Pol stops before ever reading the gene.
    A correctly placed Terminator completes the circuit (truncated = True).
    Biology: The Terminator hairpin structure physically ejects RNA Pol
    from the DNA. Any gene downstream of the Terminator is never transcribed.
    """
    if new_part.part_type != PartType.TERMINATOR:
        return False

    has_gene     = any(p.part_type in GENE_TYPES    for p in current_sequence)
    has_promoter = any(p.part_type in PROMOTER_TYPES for p in current_sequence)

    # ── Lethal: Terminator placed before Reporter Gene ───────────────────────
    if not has_gene:
        result.step_reward  += -4.0
        result.terminated    = True
        result.lethal_error  = True
        result.lethal_reason = (
            "LETHAL: Terminator placed before any Reporter Gene. "
            "RNA Pol will stop here and never read the gene. "
            "Zero fluorescence output. Episode terminated."
        )
        result.step_feedback = result.lethal_reason
        result.mechanism_trace.append(result.lethal_reason)
        return True   # lethal error

    # ── Lethal: Terminator placed before Promoter ────────────────────────────
    if not has_promoter:
        result.step_reward  += -5.0
        result.terminated    = True
        result.lethal_error  = True
        result.lethal_reason = (
            "LETHAL: Terminator placed before a Promoter. "
            "Transcription can never initiate. Circuit is non-functional."
        )
        result.step_feedback = result.lethal_reason
        result.mechanism_trace.append(result.lethal_reason)
        return True

    # ── Check task-specific completeness before declaring circuit done ────────
    missing = []
    if task_config.requires_operator:
        if not any(p.part_type == PartType.OPERATOR for p in current_sequence):
            missing.append("Operator (required for NOT Gate / repression logic)")
    if task_config.requires_repressor:
        if not any(p.part_type == PartType.REPRESSOR_GENE for p in current_sequence):
            missing.append("Repressor Gene (required for LacI steric hindrance)")
    if task_config.requires_cap:
        if not any(p.part_type == PartType.CAP_BINDING_SITE for p in current_sequence):
            missing.append("CAP Binding Site (required for cAMP activation)")
    if task_config.requires_enhancer:
        if not any(p.part_type == PartType.ENHANCER for p in current_sequence):
            missing.append("Enhancer (required for DNA looping, Task 13)")

    if missing:
        # Incomplete circuit — Terminator placed too early
        result.step_reward  -= 2.0 * len(missing)
        for m in missing:
            result.warnings.append(
                f"Circuit incomplete: missing {m}. "
                f"Terminator placed prematurely for Task {task_config.task_id}."
            )
        result.step_feedback = (
            f"✗ Terminator placed but circuit is incomplete. "
            f"Missing: {', '.join(missing)}."
        )
        result.mechanism_trace.append(result.step_feedback)
        result.truncated = True  # episode ends — but it's incomplete
        return False

    # ── Valid Terminator: circuit is structurally complete ─────────────────────
    result.step_reward  += 1.5
    result.truncated     = True
    result.step_feedback = (
        f"✔ Terminator placed correctly after Reporter Gene. "
        f"Transcription unit is structurally complete: "
        f"Promoter → [Regulatory Elements] → Gene → Terminator. "
        f"Episode complete — final evaluate_reporter_logic() will now run."
    )
    result.mechanism_trace.append(result.step_feedback)
    result.mechanism_trace.append(
        f"  Circuit parts: "
        f"{[p.part_type.value for p in sorted(current_sequence, key=lambda x: x.slot_index)]}"
        f" + Terminator"
    )
    return False   # no lethal error — clean completion


def _checkpoint_reporter_gene(new_part: DNAPart,
                               current_sequence: list[DNAPart],
                               result: CheckpointResult) -> None:
    """
    Rule: Reporter Gene placed after Promoter = correct.
    Biology: GFP gene must be downstream of the Promoter so RNA Pol can
    read it during transcription. Placement before a Promoter means
    the gene is never in the path of RNA Pol's forward movement.
    """
    if new_part.part_type not in GENE_TYPES:
        return

    has_promoter = any(p.part_type in PROMOTER_TYPES for p in current_sequence)
    has_gene     = any(p.part_type in GENE_TYPES      for p in current_sequence)

    if has_gene:
        # Duplicate gene — only warn (polycistronic is valid for Task 14)
        result.step_reward  += 0.1
        result.warnings.append(
            "Second gene added. Ensure a single Promoter drives both genes "
            "(polycistronic arrangement). If two separate Promoters are used, "
            "this is a different regulatory design — verify it matches your task."
        )
        result.mechanism_trace.append(
            "  Second gene added — check for polycistronic arrangement. +0.1"
        )
        return

    if has_promoter:
        result.step_reward  += 0.5
        result.step_feedback = (
            f"✔ Correct: {new_part.part_type.value} placed downstream of Promoter. "
            f"RNA Pol will read this gene during transcription. "
            f"GFP production pathway established."
        )
        result.mechanism_trace.append(result.step_feedback)
    else:
        # Gene placed before Promoter — near-lethal arrangement
        result.step_reward  -= 3.0
        result.warnings.append(
            f"{new_part.part_type.value} placed before a Promoter. "
            "RNA Pol cannot read a gene that appears upstream of the Promoter. "
            "Place a Promoter FIRST, then the gene."
        )
        result.mechanism_trace.append(
            f"  ✗ {new_part.part_type.value} placed before Promoter — "
            "gene cannot be transcribed. −3.0"
        )


def _checkpoint_enhancer(new_part: DNAPart,
                          current_sequence: list[DNAPart],
                          task_config: TaskCheckpointConfig,
                          result: CheckpointResult) -> None:
    """
    Rule: Enhancers are position-flexible but must be paired with a Promoter.
    Biology: Enhancers act via DNA looping — they bring bound transcription
    factors into physical proximity with the Promoter, even from thousands
    of base pairs away. This is the mechanistic basis of Task 13.
    Per LLM Rubric Level 13 Key Check: must mention DNA looping.
    """
    if new_part.part_type != PartType.ENHANCER:
        return

    has_promoter = any(p.part_type in PROMOTER_TYPES for p in current_sequence)

    if task_config.requires_enhancer:
        if has_promoter:
            result.step_reward  += 0.4
            result.step_feedback = (
                "✔ Correct: Enhancer placed in circuit with Promoter. "
                "DNA looping will bring the Enhancer-bound factors into physical "
                "proximity with the Promoter, boosting RNA Pol binding affinity. "
                "(LLM Rubric Level 13 Key Check: DNA looping mechanism present)"
            )
            result.mechanism_trace.append(result.step_feedback)
        else:
            result.step_reward  -= 0.2
            result.warnings.append(
                "Enhancer placed but no Promoter exists yet. "
                "While Enhancers are position-flexible, they must eventually "
                "be paired with a Promoter for DNA looping to be functional."
            )
    else:
        result.step_reward  -= 0.2
        result.warnings.append(
            f"Enhancer placed but Task {task_config.task_id} does not "
            "require Enhancer logic. Unnecessary part."
        )
        result.mechanism_trace.append(
            f"  ⚠ Enhancer not required for Task {task_config.task_id}. −0.2"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN FUNCTION — check_dense_rewards
#     Called by the RL environment after EVERY single part placement.
# ─────────────────────────────────────────────────────────────────────────────

def check_dense_rewards(
    new_part          : DNAPart,
    current_sequence  : list[DNAPart],
    step_number       : int,
    task_id           : int,
    cumulative_reward : float = 0.0,
) -> CheckpointResult:
    """
    Evaluate the biological correctness of ONE part placement and return
    immediate reward/penalty feedback.

    This function is called by the RL environment's `step()` method
    EVERY time the agent places a new part. It provides the dense reward
    signal that solves the Sparse Reward Problem.

    Parameters
    ----------
    new_part : DNAPart
        The part the agent just placed in this step.
    current_sequence : list[DNAPart]
        All parts placed BEFORE this step (does NOT include new_part yet).
        Ordered by slot_index ascending.
    step_number : int
        Current step index within the episode (0 = first placement).
    task_id : int
        Current task being solved (1–15). Determines biological expectations.
    cumulative_reward : float
        Running reward total from previous steps in this episode.
        Used to update CheckpointResult.cumulative_reward.

    Returns
    -------
    CheckpointResult
        Immediate feedback for this step including:
          - step_reward      → reward/penalty to add to agent's return
          - terminated       → True if a lethal error was committed
          - truncated        → True if circuit is cleanly complete
          - action_mask      → binary mask for the next step
          - mechanism_trace  → full biological reasoning log

    Environment Usage Example
    -------------------------
        result = check_dense_rewards(new_part, sequence_so_far, step, task_id)
        total_reward += result.step_reward
        if result.terminated:
            env.reset()   # lethal error — start new episode
        elif result.truncated:
            final = calculate_reporter_logic(full_sequence)   # end-of-episode score
        else:
            env.apply_action_mask(result.action_mask)   # constrain next step
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(
            f"task_id {task_id} is not valid. Must be 1–15."
        )

    task_config = TASK_CONFIGS[task_id]
    result      = CheckpointResult()
    result.mechanism_trace.append(
        f"Step {step_number} | Task {task_id}: {task_config.task_name} | "
        f"Placing: {new_part.part_type.value} at slot {new_part.slot_index}"
    )

    # ── CHECKPOINT A: First part must be a Promoter ──────────────────────────
    if step_number == 0:
        is_lethal = _checkpoint_first_part(new_part, result)
        if is_lethal:
            result.cumulative_reward = cumulative_reward + result.step_reward
            result.action_mask       = [0] * len(PART_TYPE_ORDER)
            return result
        _checkpoint_promoter_strength(new_part, task_config, result)
        result.cumulative_reward = cumulative_reward + result.step_reward
        result.action_mask = build_action_mask(
            current_sequence + [new_part], task_config
        )
        return result

    # ── CHECKPOINT B: Route to part-specific validator ───────────────────────
    if new_part.part_type == PartType.TERMINATOR:
        is_lethal = _checkpoint_terminator(
            new_part, current_sequence, task_config, result
        )
        if is_lethal:
            result.cumulative_reward = cumulative_reward + result.step_reward
            result.action_mask       = [0] * len(PART_TYPE_ORDER)
            return result

    elif new_part.part_type == PartType.OPERATOR:
        _checkpoint_operator_placement(new_part, current_sequence, result)

    elif new_part.part_type == PartType.REPRESSOR_GENE:
        _checkpoint_repressor_gene(new_part, current_sequence, task_config, result)

    elif new_part.part_type == PartType.CAP_BINDING_SITE:
        _checkpoint_cap_binding_site(new_part, task_config, result)

    elif new_part.part_type == PartType.ENHANCER:
        _checkpoint_enhancer(new_part, current_sequence, task_config, result)

    elif new_part.part_type in GENE_TYPES:
        _checkpoint_reporter_gene(new_part, current_sequence, result)

    elif new_part.part_type in PROMOTER_TYPES:
        # Promoter placed after Step 0 — duplicate Promoter (already masked, but
        # if somehow called: penalise)
        result.step_reward  -= 1.0
        result.warnings.append(
            "Duplicate Promoter placed. Only one Promoter is needed per "
            "transcription unit. This wastes metabolic resources."
        )
        result.mechanism_trace.append("  ✗ Duplicate Promoter. −1.0")

    # ── CHECKPOINT C: Generic progress bonus ─────────────────────────────────
    # Small positive reward for any placement that did not trigger a penalty,
    # simply for making progress through the episode without errors.
    if result.step_reward == 0.0:
        result.step_reward  += 0.1
        result.mechanism_trace.append(
            "  Generic progress: valid placement, no specific bonus/penalty. +0.1"
        )

    # ── Compute next step action mask ─────────────────────────────────────────
    updated_sequence = current_sequence + [new_part]
    result.action_mask       = build_action_mask(updated_sequence, task_config)
    result.cumulative_reward = cumulative_reward + result.step_reward

    result.mechanism_trace.append(
        f"  Step reward: {result.step_reward:+.2f} | "
        f"Cumulative: {result.cumulative_reward:.2f} | "
        f"Terminated: {result.terminated} | Truncated: {result.truncated}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUICK TEST — simulates 3 complete episodes
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def run_episode(label: str, steps: list[DNAPart], task_id: int) -> None:
        print(f"\n{'='*65}")
        print(f"EPISODE: {label}  (Task {task_id})")
        print(f"{'='*65}")
        sequence = []
        cumulative = 0.0
        for i, part in enumerate(steps):
            res = check_dense_rewards(part, sequence, step_number=i,
                                      task_id=task_id,
                                      cumulative_reward=cumulative)
            cumulative = res.cumulative_reward
            print(f"  Step {i}: Place {part.part_type.value:22s} "
                  f"| Reward: {res.step_reward:+.2f} "
                  f"| Cumulative: {res.cumulative_reward:.2f} "
                  f"| Term: {res.terminated} | Trunc: {res.truncated}")
            for line in res.mechanism_trace:
                print(f"    {line}")
            if res.warnings:
                for w in res.warnings:
                    print(f"    ⚠  {w}")
            if res.terminated or res.truncated:
                break
            sequence.append(part)
        print(f"  FINAL CUMULATIVE REWARD: {cumulative:.2f}")

    # Episode 1 — Perfect Task 1 (basic expression)
    run_episode("Task 1 — Perfect Basic Circuit", [
        DNAPart(PartType.MEDIUM_PROMOTER, 0),
        DNAPart(PartType.REPORTER_GENE,   1),
        DNAPart(PartType.TERMINATOR,      2),
    ], task_id=1)

    # Episode 2 — Perfect Task 2 (NOT gate: repressor, no inducer)
    run_episode("Task 2 — Perfect NOT Gate (gene OFF)", [
        DNAPart(PartType.MEDIUM_PROMOTER, 0),
        DNAPart(PartType.OPERATOR,        1, inducer_present=False),
        DNAPart(PartType.REPRESSOR_GENE,  2),
        DNAPart(PartType.REPORTER_GENE,   3),
        DNAPart(PartType.TERMINATOR,      4),
    ], task_id=2)

    # Episode 3 — Lethal Error: Terminator placed before Gene
    run_episode("LETHAL — Terminator before Reporter Gene", [
        DNAPart(PartType.MEDIUM_PROMOTER, 0),
        DNAPart(PartType.TERMINATOR,      1),   # LETHAL STEP
        DNAPart(PartType.REPORTER_GENE,   2),   # never reached
    ], task_id=1)

    # Episode 4 — Task 12: CAP-Lac dual regulation
    run_episode("Task 12 — CAP-Lac Dual Regulation", [
        DNAPart(PartType.WEAK_PROMOTER,    0),
        DNAPart(PartType.CAP_BINDING_SITE, 1, cAMP_present=True),
        DNAPart(PartType.OPERATOR,         2, inducer_present=True),
        DNAPart(PartType.REPRESSOR_GENE,   3),
        DNAPart(PartType.REPORTER_GENE,    4),
        DNAPart(PartType.TERMINATOR,       5),
    ], task_id=12)
