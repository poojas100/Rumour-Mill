import random
from uuid import uuid4
from typing import Dict, List, Any

from environment.characters import build_default_characters
from environment.ground_truth import generate_scenario
from environment.models import RumorAction, RumorObservation, RumorState
from environment.reward import calculate_final_reward, calculate_reward

from openenv.core.env_server.interfaces import Environment


ALLOWED_ACTIONS = [
    "message_character",
    "make_decision",
    "post_reddit",
    "wait",
]


class RumorMillEnv(Environment[RumorAction, RumorObservation, RumorState]):

    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty
        self.max_days = 5
        self.characters = build_default_characters()

        self._init_episode()

    # -------------------------
    # INIT / RESET
    # -------------------------
    def _init_episode(self):
        self.ground_truth = generate_scenario(difficulty=self.difficulty)
        self.current_day = 0
        self.agent_actions_history = []
        self.social_capital = 100
        self.confirmed_sources = []
        self.signal_log = []
        self.used_targets = {}

        self._state = RumorState(
            episode_id=str(uuid4()),
            step_count=0,
            current_day=self.current_day,
            max_days=self.max_days,
            social_capital=self.social_capital,
            ground_truth=self.ground_truth,
            agent_actions_history=self.agent_actions_history,
            confirmed_sources=self.confirmed_sources,
            signal_log=self.signal_log,
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> RumorObservation:
        if seed is not None:
            random.seed(seed)

        # -------- FIXED BUG --------
        avg_recent = 0
        if len(self.agent_actions_history) > 0:
            recent_rewards = [a["reward"] for a in self.agent_actions_history[-10:]]
            avg_recent = sum(recent_rewards) / max(len(recent_rewards), 1)

            if avg_recent > 20 and self.difficulty < 3:
                self.difficulty += 1
                print(f"[ENV] Difficulty increased → {self.difficulty}")

        if avg_recent < -5 and self.difficulty > 1:
            self.difficulty -= 1
            print(f"[ENV] Difficulty decreased → {self.difficulty}")

        self._init_episode()
        return self._generate_observations()

    # -------------------------
    # OBSERVATION
    # -------------------------
    def _generate_observations(
        self,
        reward: float = 0.0,
        done: bool = False,
        reward_breakdown: Dict | None = None,
        dm_response: str | None = None,
        reactions: Dict | None = None,
        ground_truth_revealed: Dict | None = None,
    ) -> RumorObservation:

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

                structured_msg = {
                    "speaker": name,
                    "content": message
                }

                if char.posts_on_reddit:
                    reddit_posts.append(structured_msg)
                else:
                    messages.append(structured_msg)

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
            reward_breakdown=reward_breakdown or {},
            done=done,
        )

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action: RumorAction | Dict) -> RumorObservation:

        if isinstance(action, dict):
            action = RumorAction(**action)

        reward = 0.0
        reward_breakdown = {}
        dm_response = None
        reactions = None
        ground_truth_revealed = None

        # -------- ACTION VALIDATION --------
        if action.type not in ALLOWED_ACTIONS:
            return self._generate_observations(
                reward=-5,
                reward_breakdown={"invalid_action": -5},
                done=False
            )

        # -------------------------
        # MESSAGE CHARACTER
        # -------------------------
        if action.type == "message_character":
            target = action.target
            question = action.content or ""

            if target not in self.characters:
                dm_response = "Invalid person."
                reward = -2
                reward_breakdown["invalid_target"] = -2
            else:
                response = self.characters[target].respond(
                    question=question,
                    ground_truth=self.ground_truth,
                    agent_reputation=self.social_capital,
                )

                dm_response = response

                # -------- anti-spam --------
                self.used_targets[target] = self.used_targets.get(target, 0) + 1
                spam_penalty = -0.5 * max(0, self.used_targets[target] - 2)

                self.confirmed_sources.append(target)

                # signal tracking
                signal_type = None
                if response:
                    if any(w in response.lower() for w in ["layoff", "cut", "bad"]):
                        signal_type = "negative"
                    elif any(w in response.lower() for w in ["good", "fine", "amazing"]):
                        signal_type = "positive"

                if signal_type:
                    self.signal_log.append({"type": signal_type, "source": target})

                base_reward, self.social_capital, reward_breakdown = calculate_reward(
                    action_type="message_character",
                    decision="",
                    target=target,
                    ground_truth=self.ground_truth,
                    current_day=self.current_day,
                    social_capital=self.social_capital,
                    action_history=self.agent_actions_history,
                    confirmed_sources=self.confirmed_sources,
                )

                reward = base_reward + spam_penalty

                reward_breakdown = {
                    "base": base_reward,
                    "spam_penalty": spam_penalty,
                }

                self.agent_actions_history.append({
                    "day": self.current_day,
                    "action": f"message {target}",
                    "target": target,
                    "reward": reward,
                })

        # -------------------------
        # DECISION
        # -------------------------
        elif action.type == "make_decision":
            decision = action.decision or ""

            base_reward, self.social_capital = calculate_reward(
                action_type="make_decision",
                decision=decision,
                target="",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
            )

            reward = base_reward
            reward_breakdown = {"decision_reward": base_reward}

            self.agent_actions_history.append({
                "day": self.current_day,
                "action": decision,
                "reward": reward,
            })

        # -------------------------
        # REDDIT
        # -------------------------
        elif action.type == "post_reddit":
            post = action.content or ""

            reactions = self._simulate_reddit_reactions(post)

            base_reward, self.social_capital = calculate_reward(
                action_type="post_anonymously_to_forum",
                decision="",
                target="",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
            )

            reward = base_reward
            reward_breakdown = {"reddit_reward": base_reward}

        # -------------------------
        # WAIT
        # -------------------------
        elif action.type == "wait":
            base_reward, self.social_capital = calculate_reward(
                action_type="wait",
                decision="",
                target="",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
            )

            reward = base_reward
            reward_breakdown = {"wait_penalty": base_reward}

        # -------------------------
        # STEP UPDATE
        # -------------------------
        self.current_day += 1
        self._state.step_count += 1

        done = (
            self.current_day >= self.max_days
            or self.social_capital <= 0
        )

        if done:
            final_reward = calculate_final_reward(
                ground_truth=self.ground_truth,
                action_history=self.agent_actions_history,
                social_capital=self.social_capital,
                confirmed_sources=self.confirmed_sources,
            )
            reward += final_reward
            reward_breakdown["final_bonus"] = final_reward
            ground_truth_revealed = self.ground_truth

        self._sync_state()

        return self._generate_observations(
            reward=reward,
            reward_breakdown=reward_breakdown,
            done=done,
            dm_response=dm_response,
            reactions=reactions,
            ground_truth_revealed=ground_truth_revealed,
        )

    # -------------------------
    # STATE SYNC
    # -------------------------
    def _sync_state(self):
        self._state.current_day = self.current_day
        self._state.social_capital = self.social_capital
        self._state.ground_truth = self.ground_truth
        self._state.agent_actions_history = self.agent_actions_history
        self._state.confirmed_sources = self.confirmed_sources
        self._state.signal_log = self.signal_log

    @property
    def state(self) -> RumorState:
        return self._state

    # -------------------------
    # REDDIT SIM
    # -------------------------
    def _simulate_reddit_reactions(self, post: str) -> Dict:
        base_upvotes = random.randint(1, 5)
        base_downvotes = random.randint(0, 3)

        if "layoff" in post.lower():
            base_upvotes += 2

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