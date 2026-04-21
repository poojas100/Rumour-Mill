from typing import Dict, List


def average_reward(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def truth_detection_accuracy(predictions: List[bool], labels: List[bool]) -> float:
    if not predictions or not labels or len(predictions) != len(labels):
        return 0.0
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    return correct / len(labels)


def source_ranking_accuracy(predicted_ranking: List[str], true_ranking: List[str]) -> float:
    if not predicted_ranking or not true_ranking or len(predicted_ranking) != len(true_ranking):
        return 0.0
    matches = sum(
        int(predicted == expected)
        for predicted, expected in zip(predicted_ranking, true_ranking)
    )
    return matches / len(true_ranking)


def summarize_run(total_reward: float, final_info: Dict) -> Dict:
    return {
        "total_reward": total_reward,
        "ground_truth_revealed": final_info.get("ground_truth_revealed"),
        "social_capital": final_info.get("social_capital"),
    }
