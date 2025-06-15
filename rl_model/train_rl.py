from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_model.train_reward import RewardRegressor

reward_model_path = "saved_reward_model.pt"
reward_tokenizer_path = "saved_tokenizer"

reward_model = RewardRegressor()
reward_model.load_state_dict(torch.load("saved_reward_model.pt", map_location="cpu"))
reward_model.eval()

reward_tokenizer = AutoTokenizer.from_pretrained(reward_tokenizer_path)

ppo_model_name = "gpt2"
ppo_tokenizer = AutoTokenizer.from_pretrained(ppo_model_name)
ppo_model = AutoModelForCausalLM.from_pretrained(ppo_model_name)
ppo_tokenizer.pad_token = ppo_tokenizer.eos_token 

prompts = ["", "", ""]
dataset = Dataset.from_dict({"prompt": prompts})

def compute_reward(texts):
    with torch.no_grad():
        inputs = reward_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        outputs = reward_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        return outputs.cpu().numpy()
    
config = PPOConfig(
    model_name=ppo_model_name,  # this just sets default values
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    optimize_cuda_cache=True,
)

# === 5. Initialize PPOTrainer ===
ppo_trainer = PPOTrainer(
    model=ppo_model,
    tokenizer=ppo_tokenizer,
    config=config,
    dataset=dataset,
)

for batch in ppo_trainer.dataloader:
    queries = batch["prompt"]
    responses = []

    for prompt in queries:
        input_ids = ppo_tokenizer.encode(prompt, return_tensors="pt")
        output = ppo_model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            pad_token_id=ppo_tokenizer.pad_token_id
        )
        response = ppo_tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append(response)

    rewards = compute_reward(responses)
    rewards = [float(r) for r in rewards]

    ppo_trainer.step(queries, responses, rewards)
    print(f"Queries: {queries}")
    print(f"Responses: {responses}")
    print(f"Rewards: {rewards}")