"""
Rumour Mill — Reward 

Fixes:
  1. reward_decision_correctness scaled down: correct decision ~10 max (was ~40)
     so it stays proportional to source consultation rewards (~3-5 pts)
  2. Acting-without-info penalty: -4 (was -10)
  3. Wrong-decision penalties halved so model can recover
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


def normalize_decision(decision: str) -> str:
    d = decision.lower()
    for canonical, aliases in DECISION_ALIASES.items():
        if any(alias in d for alias in aliases):
            return canonical
    return d

def reward_source_consultation(action_type: str, target: str, action_history: List[Dict]) -> float:
    if action_type != "message_character":
        return 0.0
    reliability = SOURCE_RELIABILITY.get(target, 0.5)
    base = reliability * 3.0  # 0.9–2.85
    times_consulted = sum(1 for a in action_history if a.get("target") == target) 
    if times_consulted >= 1:
        return round(-1.5 * times_consulted, 2)
    if target == "quiet_one":
        base += 1.0
    return round(base, 2)

def reward_epistemic_timing(
    action_type: str, current_day: int, confirmed_sources: List[str], signal_log: List[Dict]
) -> float:
    if action_type != "wait":
        return 0.0
    signals = [s["type"] for s in signal_log if s.get("type")]
    has_negative = "negative" in signals
    has_positive = "positive" in signals
    sources_count = len(confirmed_sources)
    
    # Extended episode (15 days) with contradictions: reward patient belief reconciliation
    # Contradictions planted at days 4-5 and resolved at days 8-9
    # Agent should hold through contradiction, not panic-decide at day 5
    if has_negative and has_positive:      return 4.0  # +1 for handling conflicting signals in extended context
    if has_negative and sources_count < 2 and current_day < 6: return 2.5  # +0.5 for patience through day 4-5 contradictions
    if sources_count == 0 and current_day == 0: return 0.5
    if sources_count >= 2 and current_day >= 6 and current_day <= 9: return 1.0  # Reward gathering during resolution window
    if sources_count >= 2 and current_day >= 10: return -1.0  # Penalize late decisions after resolution
    return 1.0


def reward_decision_correctness(
    action_type: str, decision: str, ground_truth: Dict, current_day: int, confirmed_sources: List[str]
) -> Tuple[float, float]:
    if action_type != "make_decision":
        return 0.0, 0.0

    decision = normalize_decision(decision)
    # Handle both old format (single event) and new timeline format
    event_type = ground_truth.get("event_type") or ground_truth.get("event", "")
    truth    = ground_truth.get("core_truth") or ground_truth.get("truth", {})

    high_quality = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
    info_quality = min(high_quality / 2.0, 1.0)

    # ── Correct decisions (max ~10-14, calibrated for 15-day episodes) ───────────────────────────────
    # In 15-day episodes, waiting through contradiction (day 5-8) and deciding correctly is harder
    # Reward timing-aware correctness: early correct = 6-7pts, late correct after contradiction = 8-9pts
    if event_type == "layoffs" and decision == "warn_team_quietly":
        # Bonus for deciding during/after resolution window (days 8-14) vs panicking day 4-5
        timing_bonus = 2.0 if current_day >= 8 else max(0, 1 - current_day / 6)
        return round(6.0 + timing_bonus + info_quality * 4.0, 2), 0.0

    if event_type == "revenue_miss" and decision == "request_budget_freeze":
        timing_bonus = 2.0 if current_day >= 6 else 0.0
        return round(6.0 + timing_bonus + info_quality * 4.0, 2), 0.0

    if event_type == "promotion_politics" and decision == "escalate_to_leadership":
        timing_bonus = 1.5 if current_day >= 9 else 0.0
        return round(5.0 + timing_bonus + info_quality * 3.0, 2), 0.0

    if decision == "wait_for_more_signals" and current_day < 8:  # extended patience window
        return 2.5, 0.0

    if decision == "ignore":
        if truth.get("happening") or truth.get("missed"):
            return -6.0, 0.0   
        return 3.0, 0.0

    # Wrong decisions (scaled for extended episodes with contradictions)
    if decision == "warn_team_quietly" and not (truth.get("happening") or truth.get("missed")):
        # Panic deciding at days 4-5 (during planted contradiction) = worse than later mistakes
        panic_penalty = 3.0 if 4 <= current_day <= 5 else 0.0
        return round(-8.0 - panic_penalty, 2), -10.0    

    if decision == "escalate_to_leadership" and current_day <= 2:
        return -4.0, -5.0     

    if high_quality == 0 and decision not in ["wait_for_more_signals", "ignore"]:
        return -4.0, 0.0 

    return 0.0, 0.0


def reward_social_preservation(action_type: str, decision: str, social_capital: float) -> float:
    if action_type == "make_decision":
        return 0.0
    if action_type in ["post_anonymously_to_forum", "post_reddit"]:
        return -1.0
    if social_capital >= 90:  return 0.5
    if social_capital < 50:   return -1.0
    return 0.0

def reward_anti_panic(action_type: str, decision: str, current_day: int, confirmed_sources: List[str]) -> float:
    if action_type != "make_decision":
        return 0.0
    decision = normalize_decision(decision)
    if current_day <= 1 and decision in ["warn_team_quietly", "escalate_to_leadership"]:
        high_quality = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
        if high_quality == 0:
            return -5.0    # was -8.0
    return 0.0

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
    """Always returns (total_reward: float, updated_social_capital: float)."""
    if signal_log is None:
        signal_log = []

    r_source             = reward_source_consultation(action_type, target, action_history)
    r_timing             = reward_epistemic_timing(action_type, current_day, confirmed_sources, signal_log)
    r_decision, sc_delta = reward_decision_correctness(action_type, decision, ground_truth, current_day, confirmed_sources)
    r_social             = reward_social_preservation(action_type, decision, social_capital)
    r_panic              = reward_anti_panic(action_type, decision, current_day, confirmed_sources)

    return round(r_source + r_timing + r_decision + r_social + r_panic, 2), social_capital + sc_delta


def calculate_final_reward(
    ground_truth: Dict, action_history: List[Dict], social_capital: float, confirmed_sources: List[str]
) -> float:
    # Handle both old single-event format and new timeline format
    if "event_type" in ground_truth:
        event = ground_truth.get("event_type", "")
        truth = ground_truth.get("core_truth", {})
    else:
        event = ground_truth.get("event", "")
        truth = ground_truth.get("truth", {})
    
    correct      = any(_is_correct(a.get("action", ""), event, truth) for a in action_history)
    good_sources = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7)
    harmful      = sum(1 for a in action_history if _is_harmful(a.get("action", ""), truth))
    never_decided = not any(
        a.get("action", "") not in ["wait", ""] and "message" not in a.get("action", "")
        for a in action_history
    )
    final = (
        0.40 * (1.0 if correct else 0.0) * 20 +
        0.25 * min(good_sources / 2.0, 1.0) * 20 +
        0.20 * max(0.0, social_capital / 100.0) * 20 +
        0.15 * max(0.0, 1.0 - harmful * 0.3) * 20 +
        (-5.0 if never_decided else 0.0)
    )
    return round(final, 2)


def _is_correct(action: str, event: str, truth: Dict) -> bool:
    a = action.lower()
    if event == "layoffs"            and "warn"     in a: return True
    if event == "revenue_miss"       and "freeze"   in a: return True
    if event == "promotion_politics" and "escalate" in a: return True
    return False

def _is_harmful(action: str, truth: Dict) -> bool:
    a = action.lower()
    return "panic" in a or "spread" in a or (
        "warn" in a and not truth.get("happening") and not truth.get("missed")
    )

def get_reward_breakdown(
    ground_truth: Dict, action_history: List[Dict], social_capital: float, confirmed_sources: List[str]
) -> Dict:
    # Handle both old single-event format and new timeline format
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