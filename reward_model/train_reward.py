from torch import nn
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
import joblib

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import USERNAME, USERNAMES

# Thoughts / TODO
# - use interaction score for label (include comments)


class CaptionLikesDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.texts = df["caption"].tolist()
        self.labels = df["sentiment_normalized"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

class RewardRegressor(nn.Module):
    def __init__(self, backbone_name="distilbert-base-uncased"):
        super().__init__()

        # Load the pretrained transformer (e.g. DistilBERT)
        self.backbone = AutoModel.from_pretrained(backbone_name)

        # Regression head on top of the [CLS] token's output
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 1)


    def forward(self, input_ids, attention_mask):
        # Run transformer model and get output
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the embedding of the [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)

        # Feed into regression head
        return self.regressor(cls_output).squeeze(1)  # shape: (batch_size,)

def load_data(USERNAMES):
    """
    Load and parse the json data into a df, just caption and likes for now

    Args:
        data_path: json data in format scraped from scraping_scripts/scrape_profile
    """ 
    records = []

    for username in USERNAMES:
        data_path = f"data/{username}.json"
        # Load the nested JSON
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        user = data.get("_metadata", {}).get("username", "unknown_user")
        followers = data.get("_metadata", {}).get("follower_count", 0)
        # Extract caption and like count from each post
        for post in data.get("posts", []):
            if post.get("caption", {}):
                records.append({
                                    # "username": user,
                                    "follower_count": followers,
                                    "like_count": post.get("like_count"),
                                    "comment_count": post.get("comment_count"),
                                    "caption": post.get("caption", {}).get("text", "")
                                })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


def train_model(model, train_loader, val_loader, device="cpu", epochs=3):
    """
    Train the neural network with MSE Loss

    Args:
        model (Type1): PyTorch model
        X_train: caption sentences
        y_train: score of each caption, based on likes and interaction data
    """ 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(y_pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader) # avg loss for one epoch

        # validation evaluation
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_function(y_pred, labels)
                valid_loss += loss.item()
            avg_val_loss = valid_loss / len(val_loader)
    
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): Trained PyTorch model
        data_loader (DataLoader): DataLoader for the eval set
        device (str): 'cpu' or 'cuda'

    Returns:
        dict: MSE, MAE, R2 scores
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f"Evaluation Results:\n  MSE: {mse:.4f}\n  MAE: {mae:.4f}\n  R²:  {r2:.4f}")
    return {"mse": mse, "mae": mae, "r2": r2}

def normalize_labels(df):
    df["like_log"] = np.log1p(df["like_count"])
    df["comment_log"] = np.log1p(df["comment_count"])
    df["follower_log"] = np.log1p(df["follower_count"].replace(0, 1))

    # Weighted log engagement per follower
    df["sentiment"] = (df["like_log"] + 0.3 * df["comment_log"])# / (df["follower_log"] + 1e-5)
    scaler = MinMaxScaler()
    df["sentiment_normalized"] = scaler.fit_transform(df["sentiment"].values.reshape(-1, 1))
    return df

def save_reward(model, tokenizer):
    # Save model weights
    model_save_path = "saved_reward_model.pt"
    torch.save(model.state_dict(), model_save_path)

    # Save tokenizer
    tokenizer_save_path = "saved_tokenizer"
    tokenizer.save_pretrained(tokenizer_save_path)

def main():
    df = load_data(USERNAMES)
    # Preview the result
    df = normalize_labels(df)

    print(df.head(10))
    print(len(df))

    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = CaptionLikesDataset(df_train, tokenizer)
    val_dataset = CaptionLikesDataset(df_val, tokenizer)
    test_dataset = CaptionLikesDataset(df_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = RewardRegressor()

    train_model(model, train_loader, val_loader, epochs=3)
    evaluate_model(model, test_loader)

    save_reward(model, tokenizer)

if __name__ == "__main__":
    main()