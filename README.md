# Multi-Armed Bandit Visualization

An interactive Streamlit application for visualizing and exploring Multi-Armed Bandit algorithms.

## Overview

This application provides an interactive visualization of several Multi-Armed Bandit algorithms, allowing users to:
- Compare different exploration-exploitation strategies
- See real-time updates of arm selection, rewards, and regret
- Adjust algorithm parameters and observe their effects

## Algorithms Implemented

- **Epsilon-Greedy**: Balances exploration and exploitation with a fixed exploration probability (ε)
- **Upper Confidence Bound (UCB)**: Uses confidence intervals to balance exploration and exploitation
- **Thompson Sampling**: A Bayesian approach using probability matching for exploration
- **Softmax**: Uses a temperature parameter to control the probability of selecting arms based on their estimated values

## Getting Started

### Prerequisites

- Python 3.6+
- pip

### Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

## Usage

1. Select an algorithm from the sidebar dropdown
2. Adjust the number of arms and trials
3. Set algorithm-specific parameters (epsilon, UCB constant, etc.)
4. Click "Start" to run the simulation
5. Use "Pause" or "Reset" to control the simulation

## Visualizations

The application displays four main visualizations:
- **True vs. Estimated Rewards**: Comparison between the actual reward values and the algorithm's estimates
- **Selection Counts**: How many times each arm was selected
- **Cumulative Regret**: Total regret over time
- **Recent Arm Selections**: Visual history of which arms were selected in recent trials

## Project Structure

- `app.py` - Main Streamlit application with UI and visualization code
- `bandit.py` - Implementation of the Multi-Armed Bandit algorithms
- `requirements.txt` - Required Python packages

## License

MIT

## References

- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
- [Introduction to Multi-Armed Bandits](https://arxiv.org/abs/1904.07272)