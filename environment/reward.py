from typing import Dict, List, Tuple

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

SOURCE_RELIABILITY = {
    "quiet_one": 0.95,
    "leaker": 0.80,
    "politician": 0.70,
    "gossip": 0.60,
    "spinner": 0.30,
}

DECISION_ALIASES = {
    "warn_team_quietly": ["warn"],
    "request_budget_freeze": ["freeze", "budget"],
    "escalate_to_leadership": ["escalate"],
    "wait_for_more_signals": ["wait"],
    "ignore": ["ignore"],
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
    confirmed_sources: List[str],
) -> Tuple[float, float, Dict]:

    reward = 0.0
    breakdown = {
        "epistemic": 0.0,
        "decision": 0.0,
        "penalty": 0.0,
        "exploration": 0.0,
    }

    updated_social_capital = social_capital
    truth = ground_truth["truth"]
    event = ground_truth["event"]

    # MESSAGE CHARACTER
    if action_type == "message_character":
        if target not in confirmed_sources:
            exploration_bonus = 2.0
        else:
            exploration_bonus = -2.0

        breakdown["exploration"] += exploration_bonus
        reward += exploration_bonus

        reliability = SOURCE_RELIABILITY.get(target, 0.5)
        epistemic_reward = reliability * 2.0

        breakdown["epistemic"] += epistemic_reward
        reward += epistemic_reward

        times_consulted = sum(
            1 for a in action_history if a.get("target") == target
        )

        if times_consulted > 2:
            penalty = -1.5 * times_consulted
            breakdown["penalty"] += penalty
            reward += penalty

    # WAIT
    if action_type == "wait":
        signals = [a.get("signal_type") for a in action_history if a.get("signal_type")]
        has_contradiction = "positive" in signals and "negative" in signals

        if has_contradiction:
            r = 2.0
        elif len(confirmed_sources) < 2 and current_day < 3:
            r = 1.5
        else:
            r = -1.0

        breakdown["epistemic"] += r
        reward += r

        if current_day == 0:
            reward += 0.3
            breakdown["epistemic"] += 0.3

    # REDDIT
    if action_type == "post_anonymously_to_forum":
        updated_social_capital -= 5
        r = 0.5
        breakdown["epistemic"] += r
        reward += r

    # DECISION
    if action_type == "make_decision":
        decision_reward, updated_social_capital = _evaluate_decision(
            decision,
            event,
            truth,
            current_day,
            updated_social_capital,
            confirmed_sources,
        )
        breakdown["decision"] += decision_reward
        reward += decision_reward

    # SOCIAL CAPITAL PRESSURE
    if updated_social_capital < 20:
        penalty = -3.0
        breakdown["penalty"] += penalty
        reward += penalty

    return reward, updated_social_capital, breakdown


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

    high_quality = sum(
        1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7
    )
    info_score = min(high_quality / 2, 1.0)

    if high_quality == 0 and decision not in ["wait_for_more_signals", "ignore"]:
        return -5.0, updated_social_capital

    if event == "layoffs" and decision == "warn_team_quietly":
        reward = 5 + (3 - current_day) + (info_score * 5)

    elif event == "revenue_miss" and decision == "request_budget_freeze":
        reward = 5 + (info_score * 5)

    elif event == "promotion_politics" and decision == "escalate_to_leadership":
        reward = 4 + (info_score * 4)

    elif decision == "wait_for_more_signals" and current_day < 3:
        reward = 1.5

    elif decision == "ignore":
        if truth.get("happening") or truth.get("missed"):
            reward = -5
        else:
            reward = 3

    else:
        reward = -4

    if decision == "escalate_to_leadership" and current_day <= 1:
        reward -= 3
        updated_social_capital -= 5

    return reward, updated_social_capital


def calculate_final_reward(
    ground_truth: Dict,
    action_history: List[Dict],
    social_capital: float,
    confirmed_sources: List[str],
) -> float:

    event = ground_truth["event"]
    truth = ground_truth["truth"]

    correct = any(
        _is_correct_decision(a.get("action", ""), event, truth)
        for a in action_history
    )

    good_sources = sum(
        1 for s in confirmed_sources if SOURCE_RELIABILITY.get(s, 0) > 0.7
    )

    harmful = sum(
        1 for a in action_history if _is_harmful(a.get("action", ""), truth)
    )

    accuracy = 1.0 if correct else 0.0
    epistemic = min(good_sources / 2, 1.0)
    social = max(0, social_capital / 100)
    harm_penalty = harmful * 0.2

    final = (
        0.4 * accuracy +
        0.3 * epistemic +
        0.2 * social +
        0.1 * (1 - harm_penalty)
    ) * 20

    return round(final, 2)


def _is_correct_decision(action: str, event: str, truth: Dict) -> bool:
    action = action.lower()
    if event == "layoffs" and "warn" in action:
        return True
    if event == "revenue_miss" and "freeze" in action:
        return True
    if event == "promotion_politics" and "escalate" in action:
        return True
    return False


def _is_harmful(action: str, truth: Dict) -> bool:
    action = action.lower()
    return "panic" in action or (
        "warn" in action and not truth.get("happening")
    )