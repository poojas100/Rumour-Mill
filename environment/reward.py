from typing import Dict, List, Tuple


def calculate_reward(decision: str, ground_truth: Dict, current_day: int, social_capital: float) -> Tuple[float, float]:
    """
    Reward based on decision accuracy
    """
    decision_lower = decision.lower()
    reward = 0.0
    updated_social_capital = social_capital
    truth = ground_truth["truth"]

    if "warn" in decision_lower:
        if truth.get("happening") or truth.get("missed"):
            if current_day <= 2:
                updated_social_capital -= 10
                reward = -5
            else:
                reward = 15
        else:
            updated_social_capital -= 20
            reward = -20

    return reward, updated_social_capital


def calculate_final_reward(ground_truth: Dict, action_history: List[Dict], social_capital: float) -> float:
    """
    End-of-week reward based on overall accuracy
    """
    accuracy_score = measure_belief_accuracy(ground_truth, action_history)
    social_score = social_capital / 100
    company_health = measure_company_outcome(action_history)

    final = (
        0.4 * accuracy_score +
        0.3 * social_score +
        0.3 * company_health
    ) * 100

    return final


def measure_belief_accuracy(ground_truth: Dict, action_history: List[Dict]) -> float:
    if not action_history:
        return 0.2

    last_action = action_history[-1]["action"].lower()
    event = ground_truth["event"]

    if event == "layoffs" and "warn" in last_action:
        return 0.9
    if event == "revenue_miss" and ("freeze" in last_action or "budget" in last_action):
        return 0.9
    if event == "promotion_politics" and "escalate" in last_action:
        return 0.8
    if "wait" in last_action:
        return 0.5
    return 0.3


def measure_company_outcome(action_history: List[Dict]) -> float:
    harmful_actions = 0
    for item in action_history:
        action = item["action"].lower()
        if "panic" in action or "spread" in action:
            harmful_actions += 1

    if harmful_actions == 0:
        return 0.9
    if harmful_actions == 1:
        return 0.6
    return 0.3
