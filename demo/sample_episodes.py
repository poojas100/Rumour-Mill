from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.models import RumorAction
from environment.rumor_env import RumorMillEnv


def run_sample_episode():
    env = RumorMillEnv()
    obs = env.reset()
    print("Initial observation:", obs.model_dump())

    scripted_actions = [
        RumorAction(type="message_character", target="quiet_one", content="What have you heard?"),
        RumorAction(type="wait"),
        RumorAction(type="make_decision", decision="warn engineering quietly"),
    ]

    for action in scripted_actions:
        obs = env.step(action)
        print("\nAction:", action.model_dump())
        print("Reward:", obs.reward)
        print("Done:", obs.done)
        print("DM Response:", obs.dm_response)
        print("Reactions:", obs.reactions)
        print("Truth Reveal:", obs.ground_truth_revealed)
        print("Observation:", obs.model_dump())
        print("State:", env.state.model_dump())
        if obs.done:
            break


if __name__ == "__main__":
    run_sample_episode()
