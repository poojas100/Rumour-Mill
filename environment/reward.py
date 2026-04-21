from typing import Dict, List, Tuple

# Structured action space - share this with your teammate
# to replace free text decisions
VALID_DECISIONS = [
    "ignore",
    "wait_for_more_signals",
    "warn_team_quietly",
    "escalate_to_leadership",
    "post_anonymously_to_forum",
    "attribute_blame_to_sales",
    "attribute_blame_to_engineering",
    "request_budget_freeze",
]

# Source ground truth accuracy (mirrors characters.py)
SOURCE_RELIABILITY = {
    "quiet_one": 0.95,
    "leaker": 0.80,
    "politician": 0.70,
    "gossip": 0.60,
    "spinner": 0.30,
}

DECISION_ALIASES = {
    "warn_team_quietly": ["warn", "quietly warn", "warn engineering", "warn team"],
    "request_budget_freeze": ["budget freeze", "freeze", "budget", "request freeze"],
    "escalate_to_leadership": ["escalate", "leadership", "report up"],
    "wait_for_more_signals": ["wait", "gather", "hold"],
    "ignore": ["ignore", "nothing", "do nothing"],
}

def normalize_decision(decision: str) -> str:
    decision_lower = decision.lower()
    for canonical, aliases in DECISION_ALIASES.items():
        if any(alias in decision_lower for alias in aliases):
            return canonical
    return decision_lower

def calculate_reward(
    action_type: str,
    decision: str,
    target: str,
    ground_truth: Dict,
    current_day: int,
    social_capital: float,
    action_history: List[Dict],
    confirmed_sources: List[str],  # sources agent has already consulted
) -> Tuple[float, float]:
    
    reward = 0.0
    updated_social_capital = social_capital
    truth = ground_truth["truth"]
    event = ground_truth["event"]

    # --- Epistemic behavior rewards (dense, every step) ---

    if action_type == "message_character":
        # Reward consulting high-reliability sources
        source_value = SOURCE_RELIABILITY.get(target, 0.5)
        reward += source_value * 3  # max +2.85 for quiet_one

        # Bonus for consulting quiet_one specifically - they rarely speak
        if target == "quiet_one":
            reward += 2

        # Penalize re-consulting same source (no new info)
        times_consulted = sum(
            1 for a in action_history
            if a.get("target") == target
        )
        if times_consulted > 1:
            reward -= 2 * times_consulted  # escalating penalty

    if action_type == "wait":
        signals = [a.get("signal_type") for a in action_history if a.get("signal_type")]
        has_any_signal = len(signals) > 0
        has_contradiction = "positive" in signals and "negative" in signals
        sources_consulted = len(confirmed_sources)

        if has_contradiction:
            reward += 3      # contradictory signals, smart to wait
        elif has_any_signal and sources_consulted < 2 and current_day < 3:
            reward += 2      # have partial info, waiting to corroborate
        elif has_any_signal and sources_consulted >= 1 and current_day < 2:
            reward += 1      # early days, cautious waiting is fine
        elif sources_consulted >= 2 and current_day >= 3:
            reward -= 2      # you have enough info, stop stalling

        # Always give tiny reward for not panic-acting on day 0
        if current_day == 0:
            reward += 0.5

    if action_type == "post_anonymously_to_forum":
        # Probing via forum post is smart but costs social capital
        updated_social_capital -= 5
        reward += 1  # small reward for creative probing

    # --- Decision rewards (sparse, high stakes) ---

    if action_type == "make_decision":
        reward, updated_social_capital = _evaluate_decision(
            decision=decision,
            event=event,
            truth=truth,
            current_day=current_day,
            social_capital=updated_social_capital,
            confirmed_sources=confirmed_sources,
        )

    return reward, updated_social_capital


def _evaluate_decision(
    decision: str,
    event: str,
    truth: Dict,
    current_day: int,
    social_capital: float,
    confirmed_sources: List[str],
) -> Tuple[float, float]:
    
    reward = 0.0
    updated_social_capital = social_capital
    decision = normalize_decision(decision)
    
    # How well-informed is this decision?
    high_reliability_sources_consulted = sum(
        1 for s in confirmed_sources
        if SOURCE_RELIABILITY.get(s, 0) > 0.7
    )
    information_quality = min(high_reliability_sources_consulted / 2, 1.0)

    # --- Correct decisions ---
    if event == "layoffs" and decision == "warn_team_quietly":
        base = 20
        timing_bonus = max(0, (5 - current_day) * 2)  # earlier correct = better
        info_bonus = information_quality * 10
        reward = base + timing_bonus + info_bonus

    elif event == "revenue_miss" and decision == "request_budget_freeze":
        base = 15
        info_bonus = information_quality * 10
        reward = base + info_bonus

    elif event == "promotion_politics" and decision == "escalate_to_leadership":
        base = 12
        reward = base + information_quality * 8

    elif decision == "wait_for_more_signals" and current_day < 3:
        # Waiting early is fine
        reward = 2

    elif decision == "ignore":
        # Ignoring a real event is bad
        if truth.get("happening") or truth.get("missed"):
            reward = -15
        else:
            reward = 5  # correctly ignoring a non-event

    # --- Wrong decisions ---
    elif decision == "warn_team_quietly" and not (
        truth.get("happening") or truth.get("missed")
    ):
        # False alarm - costs social capital
        updated_social_capital -= 25
        reward = -20

    elif decision == "escalate_to_leadership" and current_day <= 1:
        # Too early, not enough info
        updated_social_capital -= 10
        reward = -8

    # Penalize acting without good information
    if high_reliability_sources_consulted == 0 and decision not in [
        "wait_for_more_signals", "ignore"
    ]:
        reward -= 10  # never act on gossip alone

    return reward, updated_social_capital


def _signals_are_contradictory(action_history: List[Dict]) -> bool:
    """Check if agent has received conflicting signals"""
    signals = [a.get("signal_type") for a in action_history if a.get("signal_type")]
    return "positive" in signals and "negative" in signals


def calculate_final_reward(
    ground_truth: Dict,
    action_history: List[Dict],
    social_capital: float,
    confirmed_sources: List[str],
) -> float:

    event = ground_truth["event"]
    truth = ground_truth["truth"]

    # Did the agent make the right final call?
    correct_decision_made = any(
        _is_correct_decision(a.get("action", ""), event, truth)
        for a in action_history
    )

    # Did they consult quality sources before acting?
    good_sources_used = sum(
        1 for s in confirmed_sources
        if SOURCE_RELIABILITY.get(s, 0) > 0.7
    )

    # Did they avoid causing harm?
    harmful_actions = sum(
        1 for a in action_history
        if _is_harmful(a.get("action", ""), truth)
    )

    accuracy_score = 1.0 if correct_decision_made else 0.0
    epistemic_score = min(good_sources_used / 2, 1.0)
    social_score = social_capital / 100
    harm_penalty = harmful_actions * 0.2

    final = (
        0.40 * accuracy_score +
        0.25 * epistemic_score +
        0.20 * social_score +
        0.15 * (1 - harm_penalty)
    ) * 100

    return round(final, 2)


def _is_correct_decision(action: str, event: str, truth: Dict) -> bool:
    action = action.lower()
    if event == "layoffs" and "warn" in action:
        return True
    if event == "revenue_miss" and ("freeze" in action or "budget" in action):
        return True
    if event == "promotion_politics" and "escalate" in action:
        return True
    return False


def _is_harmful(action: str, truth: Dict) -> bool:
    action = action.lower()
    return "panic" in action or "spread" in action or (
        "warn" in action and not truth.get("happening")
    )