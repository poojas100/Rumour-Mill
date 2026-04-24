from environment.rumor_env import RumorMillEnv
from environment.reward import get_reward_breakdown

def task_easy():
    return RumorMillEnv(difficulty=1)

def task_medium():
    return RumorMillEnv(difficulty=2)

def task_hard():
    return RumorMillEnv(difficulty=3)


def _grade(env: RumorMillEnv, social_threshold: float) -> dict:
    breakdown = get_reward_breakdown(
        ground_truth=env.ground_truth,
        action_history=env.agent_actions_history,
        social_capital=env.social_capital,
        confirmed_sources=env.confirmed_sources,
    )

    # Composite score: correct decision + good sources + social capital
    score = (
        breakdown["accuracy"]  * 40 +
        breakdown["epistemic"] * 30 +
        breakdown["social"]    * 20 +
        breakdown["harm"]      * 10
    )

    return {
        "score":               round(score, 2),
        "success":             breakdown["correct_decision"] and env.social_capital > social_threshold,
        "correct_decision":    breakdown["correct_decision"],
        "sources_consulted":   breakdown["sources_consulted"],
        "social_capital":      env.social_capital,
        "harmful_actions":     breakdown["harmful_actions"],
        "accuracy_score":      breakdown["accuracy"],
        "epistemic_score":     breakdown["epistemic"],
    }


def grade_easy(env: RumorMillEnv):
    return _grade(env, social_threshold=80)

def grade_medium(env: RumorMillEnv):
    return _grade(env, social_threshold=70)

def grade_hard(env: RumorMillEnv):
    return _grade(env, social_threshold=60)