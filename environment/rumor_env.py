import random
from uuid import uuid4
from typing import Dict, List

from environment.characters import build_default_characters
from environment.ground_truth import generate_scenario
from environment.models import RumorAction, RumorObservation, RumorState
from environment.reward import calculate_final_reward, calculate_reward
from openenv.core.env_server.interfaces import Environment


class RumorMillEnv(Environment[RumorAction, RumorObservation, RumorState]):
    """
    The Rumor Mill: Truth discovery in noisy social networks.

    This class follows the server-side OpenEnv pattern:
    - reset() returns an Observation
    - step(action) returns an Observation
    - state exposes episode metadata
    """

    def __init__(self):
        self.ground_truth = self._generate_scenario()
        self.characters = build_default_characters()
        self.current_day = 0
        self.max_days = 5
        self.agent_actions_history = []
        self.social_capital = 100
        self._state = RumorState(
            episode_id=str(uuid4()),
            step_count=0,
            current_day=self.current_day,
            max_days=self.max_days,
            social_capital=self.social_capital,
            ground_truth=self.ground_truth,
            agent_actions_history=self.agent_actions_history,
        )

    def _generate_scenario(self) -> Dict:
        """
        Create the hidden ground truth for this episode
        """
        return generate_scenario()

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> RumorObservation:
        """
        Start new episode
        """
        if seed is not None:
            random.seed(seed)
        self.ground_truth = self._generate_scenario()
        self.current_day = 0
        self.agent_actions_history = []
        self.social_capital = 100
        self._state = RumorState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_day=self.current_day,
            max_days=self.max_days,
            social_capital=self.social_capital,
            ground_truth=self.ground_truth,
            agent_actions_history=self.agent_actions_history,
        )
        return self._generate_observations()

    def _generate_observations(
        self,
        reward: float = 0.0,
        done: bool = False,
        dm_response: str | None = None,
        reactions: Dict | None = None,
        ground_truth_revealed: Dict | None = None,
    ) -> RumorObservation:
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

        return RumorObservation(
            messages=messages,
            reddit_posts=reddit_posts,
            conversations=[],
            day=self.current_day,
            social_capital=self.social_capital,
            dm_response=dm_response,
            reactions=reactions,
            ground_truth_revealed=ground_truth_revealed,
            reward=reward,
            done=done,
        )

    def step(self, action: RumorAction | Dict) -> RumorObservation:
        """
        Agent takes action, environment responds
        """
        if isinstance(action, dict):
            action = RumorAction(**action)

        reward = 0.0
        dm_response = None
        reactions = None
        ground_truth_revealed = None

        if action.type == "message_character":
            target = action.target
            question = action.content or ""

            response = self.characters[target].respond(
                question=question,
                ground_truth=self.ground_truth,
                agent_reputation=self.social_capital,
            )

            dm_response = response
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": f"message {target}",
                    "reward": reward,
                }
            )

        elif action.type == "make_decision":
            decision = action.decision or ""
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

        elif action.type == "post_reddit":
            post = action.content or ""
            reactions = self._simulate_reddit_reactions(post)
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": f"post reddit: {post}",
                    "reward": reward,
                }
            )

        elif action.type == "wait":
            self.agent_actions_history.append(
                {
                    "day": self.current_day,
                    "action": "wait",
                    "reward": reward,
                }
            )

        self.current_day += 1
        done = self.current_day >= self.max_days
        self._state.step_count += 1

        if done:
            final_reward = calculate_final_reward(
                self.ground_truth,
                self.agent_actions_history,
                self.social_capital,
            )
            reward += final_reward
            ground_truth_revealed = self.ground_truth

        self._sync_state()

        next_obs = self._generate_observations(
            reward=reward,
            done=done,
            dm_response=dm_response,
            reactions=reactions,
            ground_truth_revealed=ground_truth_revealed,
        )
        return next_obs

    @property
    def state(self) -> RumorState:
        return self._state

    def _sync_state(self) -> None:
        self._state.current_day = self.current_day
        self._state.social_capital = self.social_capital
        self._state.ground_truth = self.ground_truth
        self._state.agent_actions_history = self.agent_actions_history

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
    print("Initial observation:", obs.model_dump())

    obs = env.step({"type": "wait"})
    print(f"After action: reward={obs.reward}, done={obs.done}")
