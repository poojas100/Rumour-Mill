from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
from trl import PPOConfig, PPOTrainer
from unsloth import FastLanguageModel

from environment.models import RumorAction
from environment.rumor_env import RumorMillEnv
from training.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    MINI_BATCH_SIZE,
    MODEL_NAME,
    TEMPERATURE,
    TOTAL_EPISODES,
)


def parse_action(action_text: str) -> dict:
    """
    Parse free-form model output into a starter action dictionary.
    """
    lowered = action_text.lower()

    if "wait" in lowered:
        return RumorAction(type="wait")

    if "reddit" in lowered or "post" in lowered:
        return RumorAction(type="post_reddit", content=action_text.strip())

    if "message" in lowered or "dm" in lowered or "ask quiet" in lowered:
        return RumorAction(
            type="message_character",
            target="quiet_one",
            content="What have you heard about Friday?",
        )

    return RumorAction(type="make_decision", decision=action_text.strip())


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

ppo_config = PPOConfig(
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
)

trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer,
)

env = RumorMillEnv()

for episode in range(TOTAL_EPISODES):
    obs = env.reset()
    done = False
    episode_rewards = []

    while not done:
        prompt = f"""
You are a mid-level employee. You received these messages:
{obs.messages}

Reddit posts:
{obs.reddit_posts}

It's day {obs.day} of 5. Your reputation: {obs.social_capital}/100

What do you do?
Options:
1. Message a specific person
2. Post on Reddit
3. Make a decision
4. Wait for more information

Choose wisely. You'll learn the truth at end of week.
"""

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        action_text = tokenizer.decode(output[0], skip_special_tokens=True)
        action = parse_action(action_text)

        next_obs = env.step(action)
        episode_rewards.append(next_obs.reward)
        trainer.step([input_ids], [output], [next_obs.reward])
        done = next_obs.done
        obs = next_obs

    print(f"Episode {episode}: Total Reward = {sum(episode_rewards)}")

    if episode % 100 == 0:
        model.save_pretrained(f"checkpoints/episode_{episode}")
