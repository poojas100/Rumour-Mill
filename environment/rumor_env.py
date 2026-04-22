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

    def __init__(self, difficulty: int = 1):
        self._same_action_streak = 0
        self._last_action_type = None
        self.difficulty = difficulty
        self.ground_truth = self._generate_scenario()
        self.characters = build_default_characters()
        self.current_day = 0
        self.max_days = 5
        self.agent_actions_history = []
        self.social_capital = 100
        self.confirmed_sources = []
        self.signal_log = []
        self.max_same_action_streak = 0
        self._last_action_type = None
        self._state = RumorState(
            episode_id=str(uuid4()),
            step_count=0,
            current_day=self.current_day,
            max_days=self.max_days,
            social_capital=self.social_capital,
            ground_truth=self.ground_truth,
            agent_actions_history=self.agent_actions_history,
            confirmed_sources=[],
            signal_log=[],
        )

    def _generate_scenario(self) -> Dict:
        return generate_scenario(difficulty=self.difficulty)

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> RumorObservation:
        """
        Start new episode
        """
        self._same_action_streak = 0
        self._last_action_type = None
        if seed is not None:
            random.seed(seed)

        if len(self.agent_actions_history) > 0:
            recent = [a["reward"] for a in self.agent_actions_history[-10:]]
            avg    = sum(recent) / max(len(recent), 1)
            if avg > 20 and self.difficulty < 3:
                self.difficulty += 1
            elif avg < -5 and self.difficulty > 1:
                self.difficulty -= 1

        if len(self.agent_actions_history) > 0:
            recent_rewards = [
                a["reward"] for a in self.agent_actions_history[-10:]
            ]
            avg_recent = sum(recent_rewards) / max(len(recent_rewards), 1)
            
            if avg_recent > 20 and self.difficulty < 3:
                self.difficulty += 1
                print(f"Difficulty increased to {self.difficulty}")
            elif avg_recent < -5 and self.difficulty > 1:
                self.difficulty -= 1
        self.ground_truth = self._generate_scenario()
        self.current_day = 0
        self.agent_actions_history = []
        self.social_capital = 100
        self.confirmed_sources = []
        self.signal_log = []
        self._state = RumorState(
            episode_id=str(uuid4()),
            step_count=0,
            current_day=self.current_day,
            max_days=self.max_days,
            social_capital=self.social_capital,
            ground_truth=self.ground_truth,
            agent_actions_history=self.agent_actions_history,
            confirmed_sources=[],
            signal_log=[],
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

    def step(self, action):
        if isinstance(action, dict):
            action = RumorAction(**action)

        dm_response = None
        reactions = None
        ground_truth_revealed = None

        # Anti-reward-hacking: penalize action loops
        if action.type == self._last_action_type:
            self._same_action_streak += 1
        else:
            self._same_action_streak = 0
            self._last_action_type = action.type

        if self._same_action_streak >= 3:
            # Force episode end with heavy penalty
            self._sync_state()
            return self._generate_observations(
                reward=-20.0,
                done=True,
                ground_truth_revealed=self.ground_truth,
            )

        if action.type == "message_character":
            target = action.target
            question = action.content or ""

            if target not in self.characters:
                dm_response = "That person doesn't exist."
            else:
                response = self.characters[target].respond(
                    question=question,
                    ground_truth=self.ground_truth,
                    agent_reputation=self.social_capital,
                )
                dm_response = response

                # Track which sources have been consulted
                self.confirmed_sources.append(target)

                # Track signal direction for contradiction detection
                if response and any(
                    w in response.lower()
                    for w in ["not good", "miss", "layoff", "cut", "happening", "bad"]
                ):
                    self.signal_log.append({"type": "negative", "source": target})
                elif response and any(
                    w in response.lower()
                    for w in ["fine", "good", "crushed", "amazing", "overblown"]
                ):
                    self.signal_log.append({"type": "positive", "source": target})

            # Reward for epistemic behavior — dense signal every step
            reward, self.social_capital = calculate_reward(
                action_type=action.type,
                decision=action.decision or "",
                target=action.target or "",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
                signal_log=self.signal_log, 
            )

            self.agent_actions_history.append({
                "day": self.current_day,
                "action": f"message {target}",
                "target": target,
                "signal_type": self.signal_log[-1]["type"] if self.signal_log else None,
                "reward": reward,
            })

        elif action.type == "make_decision":
            decision = action.decision or ""
            reward, self.social_capital = calculate_reward(
                action_type=action.type,
                decision=action.decision or "",
                target=action.target or "",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
                signal_log=self.signal_log,
            )
            self.agent_actions_history.append({
                "day": self.current_day,
                "action": decision,
                "target": None,
                "signal_type": None,
                "reward": reward,
            })

        elif action.type == "post_reddit":
            post = action.content or ""
            reactions = self._simulate_reddit_reactions(post)
            reward, self.social_capital = calculate_reward(
                action_type=action.type,
                decision=action.decision or "",
                target=action.target or "",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
                signal_log=self.signal_log,
            )
            self.agent_actions_history.append({
                "day": self.current_day,
                "action": f"post reddit: {post}",
                "target": None,
                "signal_type": None,
                "reward": reward,
            })

        elif action.type == "wait":
            reward, self.social_capital = calculate_reward(
                action_type="wait",
                decision="",
                target="",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
            )
            self.agent_actions_history.append({
                "day": self.current_day,
                "action": "wait",
                "target": None,
                "signal_type": None,
                "reward": reward,
            })

        self.current_day += 1
        done = self.current_day >= self.max_days
        self._state.step_count += 1

        if done:
            final_reward = calculate_final_reward(
                ground_truth=self.ground_truth,
                action_history=self.agent_actions_history,
                social_capital=self.social_capital,
                confirmed_sources=self.confirmed_sources,
            )
            reward += final_reward
            ground_truth_revealed = self.ground_truth

        self._sync_state()

        return self._generate_observations(
            reward=reward,
            done=done,
            dm_response=dm_response,
            reactions=reactions,
            ground_truth_revealed=ground_truth_revealed,
        )

    @property
    def state(self) -> RumorState:
        return self._state

    def _sync_state(self) -> None:
        self._state.current_day = self.current_day
        self._state.social_capital = self.social_capital
        self._state.ground_truth = self.ground_truth
        self._state.agent_actions_history = self.agent_actions_history
        self._state.confirmed_sources = self.confirmed_sources
        self._state.signal_log = self.signal_log

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
