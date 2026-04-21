import random
from typing import Dict, List, Tuple

import gym

from environment.characters import build_default_characters
from environment.ground_truth import generate_scenario
from environment.reward import calculate_final_reward, calculate_reward


class RumorMillEnv(gym.Env):
    """
    The Rumor Mill: Truth discovery in noisy social networks
    """

    def __init__(self):
        self.ground_truth = self._generate_scenario()
        self.characters = build_default_characters()
        self.current_day = 0
        self.max_days = 5
        self.agent_actions_history = []
        self.social_capital = 100

        self.observation_space = {
            "messages": List[str],
            "reddit_posts": List[str],
            "conversations": List[str],
            "day": int,
            "social_capital": float,
        }

        self.action_space = {
            "message_character": str,
            "post_reddit": str,
            "make_decision": str,
            "wait": None,
        }

    def _generate_scenario(self) -> Dict:
        """
        Create the hidden ground truth for this episode
        """
        return generate_scenario()

    def reset(self) -> Dict:
        """
        Start new episode
        """
        self.ground_truth = self._generate_scenario()
        self.current_day = 0
        self.agent_actions_history = []
        self.social_capital = 100
        return self._generate_observations()

    def _generate_observations(self) -> Dict:
        """
        Create the messages/posts the agent sees this turn
        """
        messages = []
        reddit_posts = []

        for name, char in self.characters.items():
            if char.should_speak(self.current_day):
                message = char.generate_message(
                    ground_truth=self.ground_truth,
                    day=self.current_day,
                )
                if not message:
                    continue

                if char.posts_on_reddit:
                    reddit_posts.append(message)
                else:
                    messages.append(f"{name}: {message}")

        return {
            "messages": messages,
            "reddit_posts": reddit_posts,
            "conversations": [],
            "day": self.current_day,
            "social_capital": self.social_capital,
        }

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        Agent takes action, environment responds
        """
        reward = 0.0
        info = {}

        if action["type"] == "message_character":
            target = action["target"]
            question = action["content"]

            response = self.characters[target].respond(
                question=question,
                ground_truth=self.ground_truth,
                agent_reputation=self.social_capital,
            )

            info["dm_response"] = response
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": f"message {target}",
                    "reward": reward,
                }
            )

        elif action["type"] == "make_decision":
            decision = action["decision"]
            reward, self.social_capital = calculate_reward(
                decision,
                self.ground_truth,
                self.current_day,
                self.social_capital,
            )
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": decision,
                    "reward": reward,
                }
            )

        elif action["type"] == "post_reddit":
            post = action["content"]
            reactions = self._simulate_reddit_reactions(post)
            info["reactions"] = reactions
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": f"post reddit: {post}",
                    "reward": reward,
                }
            )

        elif action["type"] == "wait":
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": "wait",
                    "reward": reward,
                }
            )

        self.current_day += 1
        done = self.current_day >= self.max_days

        if done:
            final_reward = calculate_final_reward(
                self.ground_truth,
                self.agent_actions_history,
                self.social_capital,
            )
            reward += final_reward
            info["ground_truth_revealed"] = self.ground_truth

        next_obs = self._generate_observations()
        return next_obs, reward, done, info

    def _simulate_reddit_reactions(self, post: str) -> Dict:
        base_upvotes = random.randint(1, 5)
        base_downvotes = random.randint(0, 3)

        if "layoff" in post.lower() or "engineering" in post.lower():
            base_upvotes += 3

        return {
            "upvotes": base_upvotes,
            "downvotes": base_downvotes,
            "comments": [
                "Do you have proof?",
                "This matches what I've been hearing.",
            ],
        }


if __name__ == "__main__":
    env = RumorMillEnv()
    obs = env.reset()
    print("Initial observation:", obs)

    obs, reward, done, _ = env.step({"type": "wait"})
    print(f"After action: reward={reward}, done={done}")
