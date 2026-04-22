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

def reward_source_consultation(
    action_type: str,
    target: str,
    action_history: List[Dict],
) -> float:
    """
    Rewards consulting high-reliability sources.
    Penalizes re-consulting the same source.
    Independent of decision correctness.
    """
    if action_type != "message_character":
        return 0.0

    reliability = SOURCE_RELIABILITY.get(target, 0.5)
    base = reliability * 3.0

    times_consulted = sum(
        1 for a in action_history
        if a.get("target") == target
    )
    if times_consulted >= 1:
        return -2.0 * times_consulted  # escalating penalty for repeat consults

    # Bonus for going to quiet_one — they rarely speak voluntarily
    if target == "quiet_one":
        base += 2.0

    return round(base, 2)


def reward_epistemic_timing(
    action_type: str,
    current_day: int,
    confirmed_sources: List[str],
    signal_log: List[Dict],
) -> float:
    """
    Rewards waiting when signals are unclear.
    Penalizes waiting when you have enough information.
    Independent of what the decision is.
    """
    if action_type != "wait":
        return 0.0

    signals = [s["type"] for s in signal_log if s.get("type")]
    has_negative  = "negative" in signals
    has_positive  = "positive" in signals
    sources_count = len(confirmed_sources)

    if has_negative and has_positive:
        return 3.0   # contradictory signals — smart to wait

    if has_negative and sources_count < 2 and current_day < 3:
        return 2.0   # partial info — waiting to corroborate

    if sources_count == 0 and current_day == 0:
        return 0.5   # day 0 caution is fine

    if sources_count >= 2 and current_day >= 3:
        return -2.0  # you have enough, stop stalling

    return 1.0


def reward_decision_correctness(
    action_type: str,
    decision: str,
    ground_truth: Dict,
    current_day: int,
    confirmed_sources: List[str],
) -> Tuple[float, float]:
    """
    Rewards correct decisions weighted by information quality.
    Penalizes acting without consulting reliable sources.
    Returns (reward, social_capital_delta).
    """
    if action_type != "make_decision":
        return 0.0, 0.0

    decision  = normalize_decision(decision)
    event     = ground_truth["event"]
    truth     = ground_truth["truth"]
    sc_delta  = 0.0

    high_quality = sum(
        1 for s in confirmed_sources
        if SOURCE_RELIABILITY.get(s, 0) > 0.7
    )
    info_quality = min(high_quality / 2.0, 1.0)

    # correct decisions
    if event == "layoffs" and decision == "warn_team_quietly":
        base         = 20.0
        timing_bonus = max(0, (5 - current_day) * 2)
        info_bonus   = info_quality * 10
        return round(base + timing_bonus + info_bonus, 2), 0.0

    if event == "revenue_miss" and decision == "request_budget_freeze":
        base       = 15.0
        info_bonus = info_quality * 10
        return round(base + info_bonus, 2), 0.0

    if event == "promotion_politics" and decision == "escalate_to_leadership":
        base       = 12.0
        info_bonus = info_quality * 8
        return round(base + info_bonus, 2), 0.0

    if decision == "wait_for_more_signals" and current_day < 3:
        return 2.0, 0.0

    if decision == "ignore":
        if truth.get("happening") or truth.get("missed"):
            return -15.0, 0.0
        return 5.0, 0.0  # correctly ignoring a non-event

    # wrong decisions
    if decision == "warn_team_quietly" and not (
        truth.get("happening") or truth.get("missed")
    ):
        sc_delta = -25.0
        return -20.0, sc_delta

    if decision == "escalate_to_leadership" and current_day <= 1:
        sc_delta = -10.0
        return -8.0, sc_delta

    # Acting without consulting anyone
    if high_quality == 0 and decision not in ["wait_for_more_signals", "ignore"]:
        return -10.0, 0.0

    return 0.0, 0.0


def reward_social_preservation(
    action_type: str,
    decision: str,
    social_capital: float,
) -> float:
    # Don't reward social preservation on wrong decisions
    if action_type == "make_decision":
        return 0.0  # decision correctness handles this entirely
    
    if action_type == "post_anonymously_to_forum":
        return -1.0

    if social_capital >= 90:
        return 0.5
    if social_capital < 50:
        return -1.0

    return 0.0


def reward_anti_panic(
    action_type: str,
    decision: str,
    current_day: int,
    confirmed_sources: List[str],
) -> float:
    """
    Penalizes panic behavior — acting on day 0/1 without any reliable sources.
    This is the anti-reward-hacking check.
    """
    if action_type != "make_decision":
        return 0.0

    decision = normalize_decision(decision)

    if current_day <= 1 and decision in [
        "warn_team_quietly", "escalate_to_leadership"
    ]:
        high_quality = sum(
            1 for s in confirmed_sources
            if SOURCE_RELIABILITY.get(s, 0) > 0.7
        )
        if high_quality == 0:
            return -8.0  # panic acting — no reliable source consulted

    return 0.0


# composite reward

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

    if signal_log is None:
        signal_log = []

    r_source   = reward_source_consultation(action_type, target, action_history)
    r_timing   = reward_epistemic_timing(action_type, current_day, confirmed_sources, signal_log)
    r_decision, sc_delta = reward_decision_correctness(
        action_type, decision, ground_truth, current_day, confirmed_sources
    )
    r_social   = reward_social_preservation(action_type, decision, social_capital)
    r_panic    = reward_anti_panic(action_type, decision, current_day, confirmed_sources)

    total_reward       = r_source + r_timing + r_decision + r_social + r_panic
    updated_social     = social_capital + sc_delta

    return round(total_reward, 2), updated_social


def calculate_final_reward(
    ground_truth: Dict,
    action_history: List[Dict],
    social_capital: float,
    confirmed_sources: List[str],
) -> float:

    event = ground_truth["event"]
    truth = ground_truth["truth"]

    correct = any(
        _is_correct(a.get("action", ""), event, truth)
        for a in action_history
    )

    good_sources = sum(
        1 for s in confirmed_sources
        if SOURCE_RELIABILITY.get(s, 0) > 0.7
    )

    harmful = sum(
        1 for a in action_history
        if _is_harmful(a.get("action", ""), truth)
    )

    # Timeout penalty — did the agent just wait the whole episode?
    all_waits = all(
        "wait" in a.get("action", "") or "message" in a.get("action", "")
        for a in action_history
    )
    never_decided = not any(
        a.get("action", "") not in ["wait", ""]
        and "message" not in a.get("action", "")
        for a in action_history
    )

    accuracy_score  = 1.0 if correct else 0.0
    epistemic_score = min(good_sources / 2.0, 1.0)
    social_score    = social_capital / 100.0
    harm_score      = max(0.0, 1.0 - harmful * 0.3)
    timeout_penalty = -10.0 if never_decided else 0.0

    final = (
        0.40 * accuracy_score  * 100 +
        0.25 * epistemic_score * 100 +
        0.20 * social_score    * 100 +
        0.15 * harm_score      * 100 +
        timeout_penalty
    )

    return round(final, 2)


def _is_correct(action: str, event: str, truth: Dict) -> bool:
    a = action.lower()
    if event == "layoffs"           and "warn" in a:       return True
    if event == "revenue_miss"      and "freeze" in a:     return True
    if event == "promotion_politics" and "escalate" in a:  return True
    return False


def _is_harmful(action: str, truth: Dict) -> bool:
    a = action.lower()
    return (
        "panic" in a or
        "spread" in a or
        ("warn" in a and not truth.get("happening") and not truth.get("missed"))
    )


def get_reward_breakdown(
    ground_truth: Dict,
    action_history: List[Dict],
    social_capital: float,
    confirmed_sources: List[str],
) -> Dict:
    """
    Returns individual reward components for UI display and monitoring.
    This is what you show judges — not just a single number.
    """
    event = ground_truth["event"]
    truth = ground_truth["truth"]

    correct      = any(_is_correct(a.get("action",""), event, truth) for a in action_history)
    good_sources = sum(1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s,0) > 0.7)
    harmful      = sum(1 for a in action_history if _is_harmful(a.get("action",""), truth))

    return {
        "accuracy":  1.0 if correct else 0.0,
        "epistemic": min(good_sources / 2.0, 1.0),
        "social":    social_capital / 100.0,
        "harm":      max(0.0, 1.0 - harmful * 0.3),
        "sources_consulted": confirmed_sources,
        "correct_decision":  correct,
        "harmful_actions":   harmful,
    }