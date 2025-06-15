# Social Media Caption Optimization with Reinforcement Learning

This project implements a complete pipeline for optimizing social media captions using reinforcement learning. The system learns from engagement data (likes, comments) to generate captions that maximize user engagement.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Scraping  │───▶│  Reward Model    │───▶│   RL Model      │
│                 │    │                  │    │                 │
│ • Profile data  │    │ • Multi-modal    │    │ • Policy net    │
│ • Post captions │    │ • Text + features│    │ • Value net     │
│ • Engagement    │    │ • Composite      │    │ • Caption gen   │
│   metrics       │    │   reward         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Components

### 1. Enhanced Reward Model (`reward_model/enhanced_reward.py`)
- **Multi-modal architecture**: Combines text embeddings with numerical features
- **Composite reward function**: Weighted combination of likes, comments, engagement rate, and recency
- **Advanced features**:
  - DistilBERT-based text encoder
  - Feature encoder for numerical signals
  - Model persistence and loading
  - RL-ready interface with state embeddings

### 2. RL Interface (`rl_model/rl_interface.py`)
- **Environment**: Manages caption generation as sequential decision making
- **Agent**: Policy and value networks for action selection
- **Training**: Episode-based learning with reward optimization
- **Generation**: Trained model can generate optimized captions

### 3. Data Pipeline (`scraping_scripts/`, `main.py`)
- **Scraping**: Automated data collection from social media profiles
- **Processing**: Feature extraction and reward computation
- **Integration**: Seamless pipeline from raw data to trained models

## Installation

```bash
# Install required packages
pip install torch transformers pandas scikit-learn numpy

# For scraping (if using scraping scripts)
pip install requests beautifulsoup4 selenium
```

## Usage

### Quick Start - Full Pipeline
```bash
python main.py --full-pipeline
```

### Step-by-Step Training

1. **Scrape Data**:
```bash
python main.py --scrape
```

2. **Train Enhanced Reward Model**:
```bash
python main.py --train-enhanced-reward
```

3. **Train RL Model**:
```bash
python main.py --train-rl
```

4. **Test Reward Model**:
```bash
python main.py --test-reward
```

### Individual Components

**Train reward model directly**:
```bash
python reward_model/enhanced_reward.py
```

**Test RL interface**:
```bash
python rl_model/rl_interface.py
```

## Configuration

Edit `config.py` to customize:

```python
# Users to scrape data from
USERNAMES = ["username1", "username2"]

# Model parameters
MODEL_CONFIG = {
    "backbone": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "epochs": 5
}

# Reward function weights
REWARD_CONFIG = {
    "weights": {
        "likes": 0.4,
        "comments": 0.3,
        "engagement_rate": 0.2,
        "recency": 0.1
    }
}
```

## Architecture Improvements Made

### From Basic to Enhanced Reward Model

**Before**:
- Single signal (likes only)
- Simple regression head
- No model persistence
- Limited feature engineering

**After**:
- Multi-modal inputs (text + numerical features)
- Composite reward function
- Model saving/loading
- Advanced training (early stopping, scheduling)
- RL-ready interface

### Key Enhancements

1. **Multi-Signal Reward Function**:
   - Likes, comments, engagement rate, recency
   - Weighted combination with configurable weights
   - Normalized features for stable training

2. **Better Architecture**:
   - Separate encoders for text and features
   - Dropout and regularization
   - Sigmoid output for bounded rewards

3. **RL Integration**:
   - State embedding extraction
   - Environment interface
   - Action space definition
   - Episode-based training

4. **Production Ready**:
   - Model persistence
   - Configuration management
   - Modular design
   - Command-line interface

## Model Performance

The enhanced reward model provides:
- **Better generalization**: Multi-modal inputs capture more engagement signals
- **Stable training**: Proper normalization and regularization
- **RL compatibility**: State embeddings for sequential decision making

## Next Steps for RL Model

1. **Advanced RL Algorithms**:
   - Replace simple policy gradient with PPO or A2C
   - Add experience replay
   - Implement proper advantage estimation

2. **Better Text Generation**:
   - Use transformer-based generation (GPT-style)
   - Implement beam search
   - Add diversity mechanisms

3. **Enhanced Features**:
   - Image analysis for visual content
   - Hashtag optimization
   - Timing optimization

4. **Evaluation**:
   - A/B testing framework
   - Human evaluation metrics
   - Long-term engagement tracking

## File Structure

```
├── config.py                 # Configuration settings
├── main.py                   # Main pipeline orchestrator
├── README.md                 # This file
├── data/                     # Data storage
│   ├── *.json               # Scraped user data
│   └── all_data.json        # Compiled dataset
├── reward_model/
│   ├── train_reward.py      # Basic reward model
│   ├── enhanced_reward.py   # Enhanced reward model
│   └── saved_models/        # Trained model storage
├── rl_model/
│   └── rl_interface.py      # RL environment and agent
├── scraping_scripts/
│   ├── scrape_profile.py    # Profile data scraping
│   ├── scrape_post.py       # Individual post scraping
│   └── parse_post.py        # Data parsing utilities
└── utils/                   # Utility functions
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please respect social media platforms' terms of service when scraping data.
