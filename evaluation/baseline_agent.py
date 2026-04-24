from environment.rumor_env import RumorMillEnv


def simple_baseline_policy(obs):
    """
    rule-based agent
    """

    day = obs.day

    # Day 0 → talk to best source
    if day == 0:
        return {
            "type": "message_character",
            "target": "quiet_one",
            "content": "What's happening?"
        }

    # Day 1 → second best source
    if day == 1:
        return {
            "type": "message_character",
            "target": "leaker",
            "content": "Any news?"
        }

    # Day 2 → wait (gather signals)
    if day == 2:
        return {"type": "wait"}

    # instead of always warn_team_quietly
    if day >= 3:
        if "budget" in str(obs.messages).lower():
            return {"type": "make_decision", "decision": "request_budget_freeze"}
        elif "promotion" in str(obs.messages).lower():
            return {"type": "make_decision", "decision": "escalate_to_leadership"}
        else:
            return {"type": "make_decision", "decision": "warn_team_quietly"}

    return {"type": "wait"}


def run_single_episode():
    env = RumorMillEnv()
    obs = env.reset()

    total_reward = 0
    done = False

    print("\n=== Single Episode ===")

    while not done:
        action = simple_baseline_policy(obs)
        obs = env.step(action)

        total_reward += obs.reward

        print(f"Day {obs.day}")
        print(f"Action: {action}")
        print(f"Reward: {obs.reward:.2f}")
        print(f"Social Capital: {obs.social_capital:.2f}")
        print("-" * 40)

        done = obs.done

    print(f"\nTotal Reward: {total_reward:.2f}")


# MULTIPLE EPISODES
def run_multiple_episodes(n=10):
    scores = []

    print("\n=== Running Multiple Episodes ===")

    for i in range(n):
        env = RumorMillEnv()
        obs = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = simple_baseline_policy(obs)
            obs = env.step(action)
            total_reward += obs.reward
            done = obs.done

        scores.append(total_reward)
        print(f"Episode {i+1}: {total_reward:.2f}")

    avg = sum(scores) / len(scores)

    print("\nAll Scores:", scores)
    print(f"Average Reward: {avg:.2f}")


if __name__ == "__main__":
    # Run both tests
    run_single_episode()
    run_multiple_episodes(10)