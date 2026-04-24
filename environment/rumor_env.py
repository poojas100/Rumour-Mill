import random
from uuid import uuid4
from typing import Dict, List

from environment.characters import build_default_characters
from environment.ground_truth import generate_scenario
from environment.models import RumorAction, RumorObservation, RumorState
from environment.reward import calculate_final_reward, calculate_reward
from openenv.core.env_server.interfaces import Environment


class RumorMillEnv(Environment[RumorAction, RumorObservation, RumorState]):

    def __init__(self, difficulty: int = 1):
        self._same_action_streak = 0 # Anti-reward-hacking: track how many times the agent has repeated the same action in a row
        self._last_action_type = None 
        self.difficulty = difficulty 
        self.characters = build_default_characters() 
        self.max_days = 5
        self._init_episode()

    def _init_episode(self):
        self.ground_truth = generate_scenario(difficulty=self.difficulty)
        self.current_day = 0
        self.agent_actions_history = []
        self.social_capital = 100.0
        self.confirmed_sources = []
        self.signal_log = []
        self._same_action_streak = 0
        self._last_action_type = None
        self._state = RumorState(
            episode_id=str(uuid4()),
            step_count=0, # not really used for anything but is helpful for debugging or logging
            current_day=self.current_day,
            max_days=self.max_days,
            social_capital=self.social_capital,
            ground_truth=self.ground_truth,
            agent_actions_history=self.agent_actions_history,
            confirmed_sources=[],
            signal_log=[],
        )

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> RumorObservation:
        if seed is not None:
            random.seed(seed)

        if self.agent_actions_history:
            recent = [a["reward"] for a in self.agent_actions_history[-10:]] # Look at the rewards from the last 10 actions to adjust difficulty
            avg = sum(recent) / max(len(recent), 1) # avg is reward per action over last 10 actions
            if avg > 10 and self.difficulty < 3: 
                self.difficulty += 1 # If the agent is doing really well, increase difficulty to make it more challenging 
                print(f"[ENV] Difficulty ↑ → {self.difficulty}")
            elif avg < -3 and self.difficulty > 1:
                self.difficulty -= 1 # If the agent is doing poorly, decrease difficulty to make it easier to learn
                print(f"[ENV] Difficulty ↓ → {self.difficulty}")

        self._init_episode()
        return self._generate_observations()

    def _generate_observations(
        self,
        reward: float = 0.0,
        done: bool = False,
        dm_response: str | None = None,
        reactions: Dict | None = None,
        ground_truth_revealed: Dict | None = None,
    ) -> RumorObservation:
        messages = []
        reddit_posts = []

        for name, char in self.characters.items(): # char is a Character object from characters.py
            if char.should_speak(self.current_day):
                message = char.generate_message(ground_truth=self.ground_truth, day=self.current_day)
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

    def step(self, action) -> RumorObservation:
        if isinstance(action, dict): 
            action = RumorAction(**action)

        dm_response = None
        reactions = None
        ground_truth_revealed = None
        reward = 0.0

        # Anti-reward-hacking: penalize repeating same action 3+ times
        if action.type == self._last_action_type:
            self._same_action_streak += 1
        else:
            self._same_action_streak = 0
            self._last_action_type = action.type

        if self._same_action_streak >= 3:
            self._sync_state() 
            return self._generate_observations(reward=-10.0, done=True, ground_truth_revealed=self.ground_truth) 

        if action.type == "message_character":
            target = action.target or ""
            question = action.content or ""

            if target not in self.characters:
                dm_response = f"Unknown character: {target}" 
                reward = -0.5
            else:
                response = self.characters[target].respond(
                    question=question,
                    ground_truth=self.ground_truth,
                    agent_reputation=self.social_capital,
                )
                dm_response = response
                self.confirmed_sources.append(target)

                if response:
                    lowered = response.lower() 
                    if any(w in lowered for w in ["not good", "miss", "layoff", "cut", "happening", "bad"]):
                        self.signal_log.append({"type": "negative", "source": target})
                    elif any(w in lowered for w in ["fine", "good", "crushed", "amazing", "overblown"]):
                        self.signal_log.append({"type": "positive", "source": target})

                reward, self.social_capital = calculate_reward(
                    action_type="message_character",
                    decision="",
                    target=target,
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
            reward, self.social_capital = calculate_reward(
                action_type="make_decision",
                decision=action.decision or "",
                target="",
                ground_truth=self.ground_truth,
                current_day=self.current_day,
                social_capital=self.social_capital,
                action_history=self.agent_actions_history,
                confirmed_sources=self.confirmed_sources,
                signal_log=self.signal_log,
            )
            self.agent_actions_history.append({
                "day": self.current_day,
                "action": action.decision or "",
                "target": None,
                "signal_type": None,
                "reward": reward,
            })

        elif action.type == "post_reddit":
            post = action.content or ""
            reactions = self._simulate_reddit_reactions(post)
            reward, self.social_capital = calculate_reward(
                action_type="post_reddit",
                decision="",
                target="",
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
                "target": None, "signal_type": None, "reward": reward,
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
                signal_log=self.signal_log,
            )
            self.agent_actions_history.append({
                "day": self.current_day,
                "action": "wait",
                "target": None, "signal_type": None, "reward": reward,
            })

        else:
            reward = -1.0
            self.agent_actions_history.append({
                "day": self.current_day, "action": f"invalid:{action.type}",
                "target": None, "signal_type": None, "reward": reward,
            })

        self.current_day += 1
        self._state.step_count += 1
        done = self.current_day >= self.max_days or self.social_capital <= 0

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
            reward=reward, done=done,
            dm_response=dm_response, reactions=reactions,
            ground_truth_revealed=ground_truth_revealed,
        )

    @property
    def state(self) -> RumorState:
        return self._state

    def _sync_state(self):
        self._state.current_day = self.current_day
        self._state.social_capital = self.social_capital
        self._state.ground_truth = self.ground_truth
        self._state.agent_actions_history = self.agent_actions_history
        self._state.confirmed_sources = self.confirmed_sources
        self._state.signal_log = self.signal_log

    def _simulate_reddit_reactions(self, post: str) -> Dict:
        base_up = random.randint(1, 5) # 
        base_down = random.randint(0, 3)
        if "layoff" in post.lower() or "engineering" in post.lower():
            base_up += 3
        return {"upvotes": base_up, "downvotes": base_down,
                "comments": ["Do you have proof?", "This matches what I've been hearing."]}


if __name__ == "__main__":
    print("=== Manual Agent Test ===")

    env = RumorMillEnv()
    obs = env.reset()

    actions = [
        {"type": "message_character", "target": "quiet_one", "content": "What's happening?"},
        {"type": "message_character", "target": "leaker", "content": "Any news?"},
        {"type": "wait"},
        {"type": "make_decision", "decision": "warn_team_quietly"},
    ]

    for a in actions:
        obs = env.step(a)
        print(f"Action: {a}")
        print(f"Reward: {obs.reward}")
        print(f"Social Capital: {obs.social_capital}")
        print("-" * 40)