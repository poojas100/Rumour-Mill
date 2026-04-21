from environment.rumor_env import RumorMillEnv


def run_sample_episode():
    env = RumorMillEnv()
    obs = env.reset()
    print("Initial observation:", obs)

    scripted_actions = [
        {"type": "message_character", "target": "quiet_one", "content": "What have you heard?"},
        {"type": "wait"},
        {"type": "make_decision", "decision": "warn engineering quietly"},
    ]

    for action in scripted_actions:
        obs, reward, done, info = env.step(action)
        print("\nAction:", action)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print("Observation:", obs)
        if done:
            break


if __name__ == "__main__":
    run_sample_episode()
