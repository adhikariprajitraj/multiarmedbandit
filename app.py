import streamlit as st
import pandas as pd
import plotly.express as px
import time

from bandit import MultiArmedBandit

st.set_page_config(page_title="Multi‑Armed Bandit", layout="wide")
st.title("Multi‑Armed Bandit Visualization")

# Sidebar controls
with st.sidebar:
    algorithm = st.selectbox("Algorithm",
        ['epsilon-greedy', 'ucb', 'thompson', 'softmax'])
    num_arms = st.slider("Number of Arms", 2, 10, 5)
    num_trials = st.slider("Trials", 10, 1000, 500, step=10)
    epsilon = st.slider("ε (for ε‑Greedy)", 0.0, 1.0, 0.1, 0.01) \
        if algorithm == 'epsilon-greedy' else None
    temperature = st.slider("Temperature (Softmax)", 0.01, 1.0, 0.1, 0.01) \
        if algorithm == 'softmax' else None
    ucb_c = st.slider("UCB constant", 0.1, 5.0, 2.0, 0.1) \
        if algorithm == 'ucb' else None
    speed = st.slider("Speed (steps/sec)", 1, 100, 50)

# Initialize / reset
if 'bandit' not in st.session_state or st.button("Reset"):
    st.session_state.bandit = MultiArmedBandit(
        num_arms=num_arms,
        algorithm=algorithm,
        epsilon=epsilon or 0.1,
        temperature=temperature or 0.1,
        ucb_constant=ucb_c or 2
    )

bandit: MultiArmedBandit = st.session_state.bandit

# Placeholders for charts and text
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
chart1 = st.empty()
chart2 = st.empty()
chart3 = st.empty()
history_chart = st.empty()

# Start / pause buttons
start, pause = st.columns(2)
running = st.session_state.get('running', False)

if start.button("Start"):
    st.session_state.running = True
    running = True
if pause.button("Pause"):
    st.session_state.running = False
    running = False

# Main simulation loop
while running and bandit.current_trial < num_trials:
    bandit.step()

    # Update status
    status_col1.metric("Trial", f"{bandit.current_trial}/{num_trials}")
    status_col2.metric("Cumul. Reward", f"{bandit.cumulative_reward:.2f}")
    avg = bandit.cumulative_reward / bandit.current_trial
    status_col3.metric("Avg. Reward", f"{avg:.3f}")
    total_regret = sum(bandit.regret_history)
    status_col4.metric("Total Regret", f"{total_regret:.2f}")

    # DataFrame for plots
    df = pd.DataFrame(bandit.history)

    # 1) True vs Estimated Rewards
    true_df = pd.DataFrame({
        'Arm': [f"A{i+1}" for i in range(bandit.num_arms)],
        'True Reward': bandit.true_rewards,
        'Estimated Reward': bandit.estimated_rewards
    })
    fig1 = px.bar(true_df, x='Arm', y=['True Reward', 'Estimated Reward'],
                  barmode='group', title="True vs Estimated Rewards")
    chart1.plotly_chart(fig1, use_container_width=True)

    # 2) Selection Counts
    count_df = pd.DataFrame({
        'Arm': [f"A{i+1}" for i in range(bandit.num_arms)],
        'Count': bandit.arm_counts
    })
    fig2 = px.bar(count_df, x='Arm', y='Count', title="Selection Counts")
    chart2.plotly_chart(fig2, use_container_width=True)

    # 3) Cumulative Regret over Trials
    fig3 = px.line(df, x='trial', y='cumulative_regret',
                   title="Cumulative Regret")
    chart3.plotly_chart(fig3, use_container_width=True)

    # 4) Recent Arm Selections
    history_chart.write(
        "Recent Arm Selections:  " +
        " ".join(f"**{row['selected_arm']}**" for row in bandit.history[-50:])
    )

    time.sleep(1.0 / speed)

    # Rerun so Streamlit can handle interactions
    st.rerun()

# Display the final results when the loop is complete
if bandit.current_trial >= num_trials and bandit.history:
    # Update final metrics
    status_col1.metric("Trial", f"{bandit.current_trial}/{num_trials}")
    status_col2.metric("Cumul. Reward", f"{bandit.cumulative_reward:.2f}")
    avg = bandit.cumulative_reward / bandit.current_trial
    status_col3.metric("Avg. Reward", f"{avg:.3f}")
    total_regret = sum(bandit.regret_history)
    status_col4.metric("Total Regret", f"{total_regret:.2f}")
    
    # Display final charts
    df = pd.DataFrame(bandit.history)
    
    true_df = pd.DataFrame({
        'Arm': [f"A{i+1}" for i in range(bandit.num_arms)],
        'True Reward': bandit.true_rewards,
        'Estimated Reward': bandit.estimated_rewards
    })
    fig1 = px.bar(true_df, x='Arm', y=['True Reward', 'Estimated Reward'],
                  barmode='group', title="True vs Estimated Rewards")
    chart1.plotly_chart(fig1, use_container_width=True)
    
    count_df = pd.DataFrame({
        'Arm': [f"A{i+1}" for i in range(bandit.num_arms)],
        'Count': bandit.arm_counts
    })
    fig2 = px.bar(count_df, x='Arm', y='Count', title="Selection Counts")
    chart2.plotly_chart(fig2, use_container_width=True)
    
    fig3 = px.line(df, x='trial', y='cumulative_regret',
                   title="Cumulative Regret")
    chart3.plotly_chart(fig3, use_container_width=True)
    
    # Final arm selection history
    history_chart.write(
        "Recent Arm Selections:  " +
        " ".join(f"**{row['selected_arm']}**" for row in bandit.history[-50:])
    )
