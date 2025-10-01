import streamlit as st
from video_card import video_card
from card import card

# ================== PAGE: REINFORCEMENT LEARNING ==================
def reinforcement_learning_page():
    st.title("🎮 Reinforcement Learning – Dạy AI tự học như con người")
    st.info("Từ khái niệm cơ bản đến các thuật toán nâng cao - dạy AI chơi game và giải quyết bài toán phức tạp")

    # ================== CHƯƠNG 1 ==================
    st.markdown("---")
    st.header("🌟 Chương 1: Khái Niệm Cơ Bản Reinforcement Learning")
    st.info("🎯 Mục tiêu: Hiểu bản chất RL, Markov Decision Process, các thành phần cốt lõi của hệ thống RL")

    with st.expander("Xem chi tiết: "):
        # Bài học
        col1, _, col2 = st.columns([8, 1, 8])
        with col1:
            card("Bài 1: Reinforcement Learning là gì?",
                 "Học qua tương tác với môi trường. Agent, Environment, Action, State, Reward. So sánh với Supervised và Unsupervised Learning.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Nguyên lý cơ bản của RL",
                 "Trial-and-error learning. Exploration vs Exploitation dilemma. Reward hypothesis. Ví dụ thực tế: dạy trẻ đi xe đạp, AlphaGo.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Markov Decision Process (MDP)",
                 "States, Actions, Rewards, Transitions. Markov Property: 'Future chỉ phụ thuộc Present'. Finite vs Infinite MDP examples.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Tổng thưởng và Discount Factor",
                 "Cumulative reward, Return. Discount factor γ: tại sao cần discount? Finite vs Infinite horizon. Present value concept.",
                 icon="📑",
                 color='#e8f5e9'
                 )
        with col2:
            card("Bài 5: Policy (Chiến lược)",
                 "Deterministic vs Stochastic policy. π(a|s): xác suất chọn action a ở state s. Optimal policy π*.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Value Functions",
                 "State-value function V(s), Action-value function Q(s,a). Bellman Equation trực quan. Recursive relationship.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Bellman Equations",
                 "Bellman Expectation Equation, Bellman Optimality Equation. Relationship giữa V* và Q*. Fixed point theorem.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: RL Problem Types",
                 "Prediction vs Control problems. Model-free vs Model-based. On-policy vs Off-policy. Episodic vs Continuing tasks.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        card("Lab 1: GridWorld Environment",
             "Tạo simple grid world, define MDP components, visualize policy và value functions, implement random policy evaluation.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Lab 2: FrozenLake Environment",
             "OpenAI Gym FrozenLake, khám phá environment interface, visualize MDP, analyze optimal policy manually.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 2 ==================
    st.markdown("---")
    st.header("🔄 Chương 2: Dynamic Programming Methods")
    st.info("🎯 Mục tiêu: Giải MDP khi biết model hoàn chỉnh với Policy Iteration và Value Iteration")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Lập trình động trong RL",
                 "Dynamic Programming assumptions: perfect model, finite MDP. Bootstrapping concept. Iterative improvement methods.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Policy Evaluation",
                 "Iterative Policy Evaluation algorithm. Convergence proof. Synchronous vs Asynchronous updates. Implementation tricks.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Policy Improvement",
                 "Policy Improvement Theorem. Greedy policy construction. Policy Improvement Theorem proof intuition.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Policy Iteration Algorithm",
                 "Kết hợp Policy Evaluation + Policy Improvement. Convergence guarantee. Step-by-step walkthrough với GridWorld example.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: Value Iteration Algorithm",
                 "Direct search for optimal value function. Value Iteration theorem. So sánh với Policy Iteration: speed vs simplicity.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Asynchronous DP Methods",
                 "In-place value iteration. Prioritized sweeping. Gauss-Seidel vs Jacobi methods. Real-time dynamic programming.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Generalized Policy Iteration",
                 "Framework cho most RL algorithms. Interaction giữa evaluation và improvement. Convergence intuition.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Limitations của DP",
                 "Curse of dimensionality. Perfect model assumption. Computational complexity. Transition sang model-free methods.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Projects
        card("Project 1: Solving GridWorld",
             "Implement Policy Iteration và Value Iteration từ scratch, visualize convergence, compare algorithms, analyze optimal policies.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: Inventory Management",
             "DP solution cho inventory control problem, model uncertainty demand, optimize reorder policies, visualize value function.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 3 ==================
    st.markdown("---")
    st.header("🎲 Chương 3: Monte Carlo Methods")
    st.info("🎯 Mục tiêu: Học từ experience episodes khi không biết model, First-visit và Every-visit MC")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Monte Carlo trong RL",
                 "Học từ complete episodes. Model-free learning. Law of Large Numbers. Unbiased estimates từ sampling.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: Monte Carlo Policy Evaluation",
                 "Estimate V(s) từ returns. First-visit vs Every-visit MC. Incremental mean updates. Convergence properties.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Monte Carlo Estimation of Action Values",
                 "Estimate Q(s,a) thay vì V(s). Exploring starts assumption. Importance của action-value functions cho control.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Monte Carlo Control",
                 "MC version của Generalized Policy Iteration. Policy improvement step. Exploring Starts method.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: ε-greedy Policies",
                 "Maintaining exploration trong control. ε-greedy policy improvement. Convergence với diminishing ε.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Off-policy Prediction",
                 "Importance sampling. Ordinary vs Weighted importance sampling. Target policy vs Behavior policy.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Off-policy Monte Carlo Control",
                 "Off-policy MC control algorithm. Incremental implementation. Variance issues với importance sampling.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: MC vs DP Comparison",
                 "Advantages: model-free, focus on important states. Disadvantages: high variance, episode requirement.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Projects
        card("Project 1: Blackjack với MC",
             "Classic Blackjack environment, implement First-visit MC prediction, estimate state values, visualize learned policy.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: MC Control cho GridWorld",
             "Compare MC Control vs Policy Iteration, analyze convergence rates, study effect of exploration parameters.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 4 ==================
    st.markdown("---")
    st.header("⚡ Chương 4: Temporal Difference Learning")
    st.info("🎯 Mục tiêu: Combine MC và DP advantages với TD learning, Q-Learning và SARSA")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Temporal Difference Learning",
                 "Bootstrap + Sampling. TD(0) algorithm. TD error concept. So sánh với MC và DP methods.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: TD Prediction Algorithm",
                 "TD(0) for policy evaluation. Step-size parameter α. Online learning vs batch methods. Convergence properties.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Q-Learning từ A-Z",
                 "Off-policy TD control. Q-Learning algorithm derivation. Optimal action-value function learning. Implementation details.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Q-Learning Implementation",
                 "Q-table representation. Action selection strategies. Learning rate schedules. Exploration decay strategies.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: SARSA Algorithm",
                 "On-policy TD control. SARSA vs Q-Learning differences. Expected SARSA variant. Policy dependency.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Q-Learning vs SARSA Comparison",
                 "Off-policy vs On-policy learning. 'Tu Tiên' analogy: SARSA conservative, Q-Learning aggressive. Cliff Walking example.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: N-step TD Methods",
                 "N-step returns. TD(λ) introduction. Eligibility traces concept. Bias-variance tradeoff trong n-step methods.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: Function Approximation Preview",
                 "Limitations của tabular methods. Linear function approximation. Neural networks cho value functions. Transition sang Deep RL.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Lab Projects
        card("Lab 1: Q-Learning CartPole",
             "Discretize continuous state space, implement Q-Learning, balance CartPole, hyperparameter tuning, convergence analysis.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Lab 2: SARSA CartPole Implementation",
             "SARSA algorithm cho CartPole, compare với Q-Learning performance, analyze on-policy vs off-policy behavior.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    # ================== CHƯƠNG 5 ==================
    st.markdown("---")
    st.header("🤖 Chương 5: Deep Reinforcement Learning")
    st.info("🎯 Mục tiêu: Kết hợp Deep Learning với RL - DQN, Policy Gradients, Actor-Critic methods")

    with st.expander("Xem chi tiết:"):
        col1, _, col2 = st.columns([8, 1, 8])

        with col1:
            card("Bài 1: Deep Q-Networks (DQN)",
                 "Neural networks cho Q-function approximation. Experience replay buffer. Target network concept. DQN algorithm overview.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 2: DQN Improvements",
                 "Double DQN, Dueling DQN, Prioritized Experience Replay. Rainbow DQN overview. Addressing overestimation bias.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 3: Policy Gradient Methods",
                 "Direct policy optimization. REINFORCE algorithm. Policy gradient theorem. Baseline để reduce variance.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 4: Actor-Critic Methods",
                 "Combine value estimation và policy optimization. Advantage function. A2C (Advantage Actor-Critic) algorithm.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        with col2:
            card("Bài 5: Proximal Policy Optimization (PPO)",
                 "Trust region methods intuition. Clipped objective function. PPO-Clip algorithm. Why PPO is popular?",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 6: Soft Actor-Critic (SAC)",
                 "Maximum entropy RL framework. Continuous action spaces. Temperature parameter. SAC algorithm components.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 7: Multi-Agent RL",
                 "Independent learners, centralized training/decentralized execution. Nash equilibrium concepts. Cooperation vs competition.",
                 icon="📑",
                 color='#e8f5e9'
                 )
            card("Bài 8: RL Applications & Future",
                 "Game playing (AlphaGo, OpenAI Five), robotics, autonomous vehicles, recommendation systems. Challenges và future directions.",
                 icon="📑",
                 color='#e8f5e9'
                 )

        # Major Projects
        card("Project 1: DQN Flappy Bird",
             "Train DQN agent chơi Flappy Bird, implement experience replay, target network, hyperparameter tuning, visualize learning process.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 2: PPO Car Racing",
             "Train PPO agent cho CarRacing-v0, continuous action space, CNN feature extraction, reward shaping, performance analysis.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        card("Project 3: SAC Walker2d",
             "Train SAC agent học đi bộ trong Walker2d environment, continuous control, compare với PPO, locomotion analysis.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

        # Final Project
        card("Capstone: Multi-Environment RL Benchmark",
             "So sánh multiple algorithms (DQN, PPO, SAC) trên various environments, implement evaluation metrics, create RL agent comparison dashboard.",
             icon="📋",
             color='linear-gradient(135deg, #fbc2eb, #a6c1ee)'
             )

    st.markdown("---")
    st.success("👉 Hoàn thành Reinforcement Learning track, bạn sẽ hiểu sâu về RL algorithms, có thể train AI agents giải quyết complex tasks và develop cutting-edge RL applications!")