from config import USERNAMES
from scraping_scripts import scrape_profile
from reward_model import train_reward
import json
import os
import argparse


def compile_data(usernames=USERNAMES):
    """Compile data from multiple users into a single file"""
    all_posts = []

    for username in usernames:
        file_path = f"data/{username}.json"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
                all_posts.extend(user_data)
        else:
            print(f"Warning: Data file not found for {username}")

    if all_posts:
        with open("data/all_data.json", "w", encoding="utf-8") as f:
            json.dump(all_posts, f, indent=2, ensure_ascii=False)
        print(f"Compiled {len(all_posts)} posts from {len(usernames)} users")
    else:
        print("No data found to compile")


def scrape_data():
    """Scrape data for all configured users"""
    print("Scraping data for users:", USERNAMES)
    for username in USERNAMES:
        print(f"Scraping data for {username}...")
        scrape_profile.run_scraper(username)
    
    compile_data(USERNAMES)


def train_reward_func():
    """Train the basic reward model"""
    print("Training basic reward model...")
    train_reward.main()


def train_rl_model():
    return


def test_reward_model():
    return


def full_pipeline():
    """Run the complete pipeline: scrape -> train reward -> train RL"""
    print("Running full pipeline...")
    
    # Step 1: Scrape data
    scrape_data()
    
    # Step 2: Train enhanced reward model
    train_reward_func()
    
    # Step 3: Train RL model
    train_rl_model()
    
    print("Full pipeline completed!")


def main():
    parser = argparse.ArgumentParser(description="Social Media Caption Optimization Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Scrape data for all users")
    parser.add_argument("--compile", action="store_true", help="Compile data for all users")
    parser.add_argument("--train-reward", action="store_true", help="Train reward model")
    parser.add_argument("--train-rl", action="store_true", help="Train RL model")
    parser.add_argument("--test-reward", action="store_true", help="Test reward model")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete pipeline")
    
    args = parser.parse_args()
    
    if args.scrape:
        scrape_data()
    elif args.compile:
        compile_data()
    elif args.train_reward:
        train_reward_func()
    elif args.train_rl:
        train_rl_model()
    elif args.test_reward:
        test_reward_model()
    elif args.full_pipeline:
        full_pipeline()
    else:
        # Default behavior - run full pipeline
        print("No specific command provided. Running full pipeline...")
        print("Use --help to see available options")
        full_pipeline()


if __name__ == "__main__":
    main()
