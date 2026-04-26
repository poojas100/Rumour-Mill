"""
Rumour Mill — Reward Function
All rewards normalized to [-1.0, +1.0] as per mentor feedback.

Normalization approach:
  - Step rewards:    raw values divided by their natural max, clamped to [-1, 1]
  - Final reward:    weighted sum already 0-1, shifted to [-1, 1]
  - All clipping:    hard clamp at the end of calculate_reward as safety net

Supports all 6+ scenarios via SCENARIO_CORRECT_DECISION lookup.
"""
from typing import Dict, List, Tuple

SOURCE_RELIABILITY = {
    "quiet_one":  0.95,
    "leaker":     0.80,
    "politician": 0.70,
    "gossip":     0.60,
    "spinner":    0.30,
}

DECISION_ALIASES = {
    "warn_team_quietly":      ["warn", "quietly", "warn engineering", "warn team", "alert", "notify"],
    "request_budget_freeze":  ["budget freeze", "freeze", "budget", "request freeze"],
    "escalate_to_leadership": ["escalate", "leadership", "report up", "tell leadership"],
    "wait_for_more_signals":  ["wait", "gather", "hold", "more signals"],
    "ignore":                 ["ignore", "nothing", "do nothing"],
}

# Correct decision per scenario — add new scenarios here only
SCENARIO_CORRECT_DECISION = {
    "layoffs":             "warn_team_quietly",
    "revenue_miss":        "request_budget_freeze",
    "promotion_politics":  "escalate_to_leadership",
    "acquisition":         "escalate_to_leadership",
    "product_launch_fail": "warn_team_quietly",
    "leadership_change":   "escalate_to_leadership",
    "election_fraud":      "escalate_to_leadership",
    "candidate_scandal":   "warn_team_quietly",
    "athlete_trade":       "warn_team_quietly",
    "doping_coverup":      "escalate_to_leadership",
    "rug_pull":            "warn_team_quietly",
    "exchange_insolvency": "warn_team_quietly",
    "research_fraud":      "escalate_to_leadership",
    "environmental_coverup": "escalate_to_leadership",
    "game_shutdown":       "warn_team_quietly",
}


def normalize_decision(decision: str) -> str:
    d = decision.lower()
    for canonical, aliases in DECISION_ALIASES.items():
        if any(alias in d for alias in aliases):
            return canonical
    return d


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────
# INDEPENDENT REWARD COMPONENTS  (each returns [-1, +1])
# ─────────────────────────────────────────────────────────────

def reward_source_consultation(action_type: str, target: str, action_history: List[Dict]) -> float:
    """
    Rewards consulting high-reliability sources for the first time.
    Penalizes re-consulting same source.
    Range: [-1, +1]
    """
    if action_type != "message_character":
        return 0.0

    reliability     = SOURCE_RELIABILITY.get(target, 0.5)
    times_consulted = sum(1 for a in action_history if a.get("target") == target)

    if times_consulted >= 1:
        # Repeat consultation — penalize, max -1.0
        raw = -0.5 * times_consulted
        return _clamp(raw)

    # First time — reward proportional to reliability (0.3 → 0.30, 0.95 → 1.0 with quiet_one bonus)
    base = reliability  # already 0-1 range
    if target == "quiet_one":
        base = min(base + 0.05, 1.0)  # small bonus for seeking out quiet sources

    return round(_clamp(base), 3)


def reward_epistemic_timing(
    action_type: str, current_day: int, confirmed_sources: List[str], signal_log: List[Dict]
) -> float:
    """
    Rewards waiting when signals conflict or info is incomplete.
    Penalizes stalling when enough info exists.
    Range: [-1, +1]
    """
    if action_type != "wait":
        return 0.0

    signals       = [s["type"] for s in signal_log if s.get("type")]
    has_negative  = "negative" in signals
    has_positive  = "positive" in signals
    sources_count = len(confirmed_sources)

    if has_negative and has_positive:                           return  1.0   # contradictory — smart to wait
    if has_negative and sources_count < 2 and current_day < 6: return  0.6   # partial info, corroborate
    if sources_count == 0 and current_day == 0:                return  0.2   # day-0 caution
    if sources_count >= 2 and 6 <= current_day <= 9:           return  0.3   # gather during resolution window
    if sources_count >= 2 and current_day >= 10:               return -0.4   # stalling too long
    return 0.3  # neutral wait


def reward_decision_correctness(
    action_type: str, decision: str, ground_truth: Dict, current_day: int, confirmed_sources: List[str]
) -> Tuple[float, float]:
    """
    Rewards making the correct decision with good information.
    Penalizes wrong or premature decisions.
    Range: [-1, +1], sc_delta is social capital change (not normalized — env units)
    """
    if action_type != "make_decision":
        return 0.0, 0.0

    decision         = normalize_decision(decision)
    event_type       = ground_truth.get("event_type") or ground_truth.get("event", "")
    truth            = ground_truth.get("core_truth") or ground_truth.get("truth", {})
    correct_decision = SCENARIO_CORRECT_DECISION.get(event_type)

    high_quality = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
    info_quality = min(high_quality / 2.0, 1.0)  # 0.0 to 1.0

    # ── CORRECT DECISION ─────────────────────────────────────
    if correct_decision and decision == correct_decision:
        # Base reward + timing bonus + info quality bonus, all in [0,1] space
        timing_bonus = 0.2 if current_day >= 8 else max(0.0, 0.1 - current_day * 0.01)
        base         = 0.5 + timing_bonus + info_quality * 0.3   # 0.5 to 1.0
        return round(_clamp(base), 3), 0.0

    # ── WAIT FOR MORE SIGNALS (reasonable early choice) ───────
    if decision == "wait_for_more_signals" and current_day < 8:
        return 0.3, 0.0

    # ── IGNORE ────────────────────────────────────────────────
    if decision == "ignore":
        something_real = (
            truth.get("happening") or truth.get("missed") or
            truth.get("failing")   or truth.get("leaving") or
            truth.get("rug_pull")  or truth.get("insolvent") or
            truth.get("fraud")     or truth.get("contamination") or
            truth.get("shutdown")  or truth.get("systematic")
        )
        return (-0.6, 0.0) if something_real else (0.4, 0.0)

    # ── WRONG / PREMATURE DECISIONS ───────────────────────────
    # Panic-decide during contradiction window (days 4-5) is worst
    panic_day = 4 <= current_day <= 5
    if decision == "warn_team_quietly" and event_type not in [
        "layoffs", "product_launch_fail", "rug_pull",
        "exchange_insolvency", "candidate_scandal", "game_shutdown"
    ]:
        penalty = -0.9 if panic_day else -0.6
        return penalty, -10.0   # also hits social capital

    if decision == "escalate_to_leadership" and current_day <= 2:
        return -0.5, -5.0

    if high_quality == 0 and decision not in ["wait_for_more_signals", "ignore"]:
        return -0.5, 0.0   # acted with no reliable info

    return 0.0, 0.0


def reward_social_preservation(action_type: str, decision: str, social_capital: float) -> float:
    """
    Small reward for maintaining social capital, small penalty for burning it.
    Range: [-0.2, +0.2] — intentionally narrow, supporting signal only
    """
    if action_type == "make_decision":
        return 0.0
    if action_type in ["post_anonymously_to_forum", "post_reddit"]:
        return -0.15
    if social_capital >= 90:
        return 0.1
    if social_capital < 50:
        return -0.2
    return 0.0


def reward_anti_panic(action_type: str, decision: str, current_day: int, confirmed_sources: List[str]) -> float:
    """
    Penalizes decisive action on day 0-1 with no reliable sources.
    Anti-reward-hacking check.
    Range: [-1, 0]
    """
    if action_type != "make_decision":
        return 0.0
    decision = normalize_decision(decision)
    if current_day <= 1 and decision in ["warn_team_quietly", "escalate_to_leadership"]:
        high_quality = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
        if high_quality == 0:
            return -0.8   # strong penalty — don't act without any reliable info
    return 0.0


# ─────────────────────────────────────────────────────────────
# COMPOSITE REWARD
# ─────────────────────────────────────────────────────────────

def calculate_reward(
    action_type: str,
    decision: str,
    target: str,
    ground_truth: Dict,
    current_day: int,
    social_capital: float,
    action_history: List[Dict],
    confirmed_sources: List[str],
    signal_log: List[Dict] = None,
) -> Tuple[float, float]:
    """
    Returns (total_reward: float in [-1,1], updated_social_capital: float).
    """
    if signal_log is None:
        signal_log = []

    r_source             = reward_source_consultation(action_type, target, action_history)
    r_timing             = reward_epistemic_timing(action_type, current_day, confirmed_sources, signal_log)
    r_decision, sc_delta = reward_decision_correctness(action_type, decision, ground_truth, current_day, confirmed_sources)
    r_social             = reward_social_preservation(action_type, decision, social_capital)
    r_panic              = reward_anti_panic(action_type, decision, current_day, confirmed_sources)

    raw = r_source + r_timing + r_decision + r_social + r_panic

    # Weighted combination, then clamp to [-1, +1]
    # Components are already mostly in [-1,1] but can stack — clamp enforces hard bound
    n_fired = sum([
        r_source != 0,
        r_timing != 0, 
        r_decision != 0,
        r_social != 0,
        r_panic != 0,
    ])
    divisor = max(1.0, n_fired * 0.6)  # less dilution for single signals
    total   = _clamp(raw / divisor)

    return round(total, 4), social_capital + sc_delta


# ─────────────────────────────────────────────────────────────
# FINAL EPISODE REWARD  ([-1, +1])
# ─────────────────────────────────────────────────────────────

def calculate_final_reward(
    ground_truth: Dict, action_history: List[Dict], social_capital: float, confirmed_sources: List[str]
) -> float:
    """
    End-of-episode bonus. Range: [-1, +1].
    Weighted sum of accuracy, epistemic quality, social preservation, harm avoidance.
    """
    if "event_type" in ground_truth:
        event = ground_truth.get("event_type", "")
        truth = ground_truth.get("core_truth", {})
    else:
        event = ground_truth.get("event", "")
        truth = ground_truth.get("truth", {})

    correct       = any(_is_correct(a.get("action", ""), event, truth) for a in action_history)
    good_sources  = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
    harmful       = sum(1 for a in action_history if _is_harmful(a.get("action", ""), truth))
    never_decided = not any(
        a.get("action", "") not in ["wait", ""] and "message" not in a.get("action", "")
        for a in action_history
    )

    accuracy_score  = 1.0 if correct else 0.0
    epistemic_score = min(good_sources / 2.0, 1.0)
    social_score    = max(0.0, social_capital / 100.0)
    harm_score      = max(0.0, 1.0 - harmful * 0.3)

    # Weighted sum → [0, 1] → shift to [-1, +1]
    raw = (
        0.40 * accuracy_score +
        0.25 * epistemic_score +
        0.20 * social_score +
        0.15 * harm_score
    )

    # Shift: 0→-1, 0.5→0, 1→+1
    shifted = (raw * 2.0) - 1.0

    if never_decided:
        shifted -= 0.3   # penalty for never making a decision

    return round(_clamp(shifted), 4)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _is_correct(action: str, event: str, truth: Dict) -> bool:
    a = action.lower()
    correct_map = {
        "layoffs":              "warn"     in a,
        "revenue_miss":         "freeze"   in a,
        "promotion_politics":   "escalate" in a,
        "acquisition":          "escalate" in a,
        "product_launch_fail":  "warn"     in a,
        "leadership_change":    "escalate" in a,
        "election_fraud":       "escalate" in a,
        "candidate_scandal":    "warn"     in a,
        "athlete_trade":        "warn"     in a,
        "doping_coverup":       "escalate" in a,
        "rug_pull":             "warn"     in a,
        "exchange_insolvency":  "warn"     in a,
        "research_fraud":       "escalate" in a,
        "environmental_coverup":"escalate" in a,
        "game_shutdown":        "warn"     in a,
    }
    return correct_map.get(event, False)


def _is_harmful(action: str, truth: Dict) -> bool:
    a = action.lower()
    return "panic" in a or "spread" in a or (
        "warn" in a
        and not truth.get("happening")
        and not truth.get("missed")
        and not truth.get("failing")
        and not truth.get("rug_pull")
        and not truth.get("insolvent")
    )


def get_reward_breakdown(
    ground_truth: Dict, action_history: List[Dict], social_capital: float, confirmed_sources: List[str]
) -> Dict:
    if "event_type" in ground_truth:
        event = ground_truth.get("event_type", "")
        truth = ground_truth.get("core_truth", {})
    else:
        event = ground_truth.get("event", "")
        truth = ground_truth.get("truth", {})

    correct      = any(_is_correct(a.get("action", ""), event, truth) for a in action_history)
    good_sources = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
    harmful      = sum(1 for a in action_history if _is_harmful(a.get("action", ""), truth))

    return {
        "accuracy":          1.0 if correct else 0.0,
        "epistemic":         min(good_sources / 2.0, 1.0),
        "social":            social_capital / 100.0,
        "harm":              max(0.0, 1.0 - harmful * 0.3),
        "sources_consulted": confirmed_sources,
        "correct_decision":  correct,
        "harmful_actions":   harmful,
    }


# ─────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Reward Sanity Check ===")
    print("All values should be in [-1.0, +1.0]\n")

    dummy_gt = {"event_type": "layoffs", "core_truth": {"happening": True, "teams": ["Engineering"]}}

    tests = [
        ("message quiet_one first time",  "message_character", "", "quiet_one", [], []),
        ("message quiet_one second time", "message_character", "", "quiet_one", [{"target": "quiet_one"}], []),
        ("wait with contradictory signals","wait", "", "", [], [{"type":"negative"},{"type":"positive"}]),
        ("correct decision with info",    "make_decision", "warn_team_quietly", "", ["quiet_one", "leaker"], []),
        ("wrong decision no info",        "make_decision", "ignore", "", [], []),
        ("panic decision day 0",          "make_decision", "warn_team_quietly", "", [], []),
    ]

    for label, action_type, decision, target, confirmed, signal_log in tests:
        r, sc = calculate_reward(
            action_type=action_type, decision=decision, target=target,
            ground_truth=dummy_gt, current_day=0,
            social_capital=100.0, action_history=[{"target": t} for t in confirmed],
            confirmed_sources=confirmed, signal_log=signal_log,
        )
        in_range = "✅" if -1.0 <= r <= 1.0 else "❌ OUT OF RANGE"
        print(f"  {in_range}  {label:40s} → {r:+.4f}")

    print("\n=== Final Reward Check ===")
    test_histories = [
        ("correct decision + good sources", [{"action": "warn"}], 100.0, ["quiet_one", "leaker"]),
        ("no decision made",               [{"action": "wait"},{"action": "wait"}], 100.0, []),
        ("wrong decision",                 [{"action": "ignore"}], 50.0, []),
    ]
    for label, history, sc, sources in test_histories:
        r = calculate_final_reward(dummy_gt, history, sc, sources)
        in_range = "✅" if -1.0 <= r <= 1.0 else "❌ OUT OF RANGE"
        print(f"  {in_range}  {label:40s} → {r:+.4f}")