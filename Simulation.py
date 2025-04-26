import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# 1. Trading Environment
class TradingEnv(gym.Env):
    """
    CSV must include:
      - Mandatory: "Close", "MA5", "MA20", "RSI", "MACD"
      - Optional: "Predicted_Next_Close", "Sentiment_Score"
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path,
                 initial_balance=10000,
                 stop_loss_multiplier=2,
                 window=5,
                 features_to_use=None):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.num_data = len(self.data)
        self.initial_balance = initial_balance
        self.stop_loss_multiplier = stop_loss_multiplier
        self.window = window

        # Default features
        if features_to_use is None:
            features_to_use = ["Close", "MA5", "MA20", "RSI", "MACD"]
        # Verify column names
        missing = [f for f in features_to_use if f not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.features_to_use = features_to_use

        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.data[self.features_to_use])

        # Action/Observation spaces
        self.action_space = spaces.Discrete(3)  # 0=Hold,1=Buy,2=Sell
        obs_dim = len(self.features_to_use) + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        self.reset()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        feats = row[self.features_to_use].values.astype(np.float32)
        scaled = self.scaler.transform(feats.reshape(1, -1)).flatten()
        return np.concatenate([scaled, [self.position]], axis=0).astype(np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.position = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.done = False
        self.equity_history = [self.initial_balance]
        return self._get_state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset().")

        prev_price = self.data.iloc[self.current_step]["Close"]
        prev_equity = self.balance + self.shares * prev_price

        # Advance time
        self.current_step += 1
        if self.current_step >= self.num_data - 1:
            self.done = True
        current_price = self.data.iloc[self.current_step]["Close"]

        # BUY
        if action == 1 and self.shares == 0 and self.balance >= current_price:
            self.shares = 1
            self.position = 1
            self.entry_price = current_price
            self.balance -= current_price
        # SELL
        elif action == 2 and self.shares > 0:
            self.balance += current_price * self.shares
            self.shares = 0
            self.position = 0
            self.entry_price = 0.0

        # STOP-LOSS
        if self.shares > 0:
            start = max(0, self.current_step - self.window + 1)
            window_prices = self.data.iloc[start:self.current_step+1]["Close"].values
            atr = np.mean(np.abs(np.diff(window_prices))) if len(window_prices) > 1 else 0.0
            if current_price < self.entry_price - self.stop_loss_multiplier * atr:
                self.balance += current_price * self.shares
                self.shares = 0
                self.position = 0
                self.entry_price = 0.0

        # Compute reward
        current_equity = self.balance + self.shares * current_price
        reward = current_equity - prev_equity
        self.equity_history.append(current_equity)

        return self._get_state(), reward, self.done, {}

    def render(self, mode='human'):
        cp = self.data.iloc[self.current_step]["Close"]
        eq = self.balance + self.shares * cp
        print(f"Step {self.current_step} | Close={cp:.2f} | "
              f"Balance={self.balance:.2f} | Shares={self.shares} | Equity={eq:.2f}")


# 2. PPO Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


# 3. PPO Agent
class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, clip_eps=0.2, epochs=10, batch_size=64):
        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps
        self.epochs, self.batch_size = epochs, batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(input_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

    def select_action(self, state):
        st = torch.FloatTensor(state).to(self.device)
        probs, val = self.ac(st)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a).item(), val.item()

    def compute_gae(self, rewards, values, dones):
        advs, gae = [], 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = (rewards[t]
                     + self.gamma * (1 - dones[t]) * values[t+1]
                     - values[t])
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
        rets = [a + v for a, v in zip(advs, values[:-1])]
        return advs, rets

    def update(self, traj):
        S = torch.FloatTensor(traj['states']).to(self.device)
        A = torch.LongTensor(traj['actions']).to(self.device)
        old_lp = torch.FloatTensor(traj['log_probs']).to(self.device)
        R, D, V = traj['rewards'], traj['dones'], traj['values']

        advs, rets = self.compute_gae(R, V, D)
        advs = torch.FloatTensor(advs).to(self.device)
        rets = torch.FloatTensor(rets).to(self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        N = S.size(0)
        for _ in range(self.epochs):
            idx = torch.randperm(N)
            for i in range(0, N, self.batch_size):
                b = idx[i:i+self.batch_size]
                probs, vals = self.ac(S[b])
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(A[b])
                ratio = (new_lp - old_lp[b]).exp()
                s1 = ratio * advs[b]
                s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advs[b]
                a_loss = -torch.min(s1, s2).mean()
                c_loss = nn.MSELoss()(vals.squeeze(-1), rets[b])
                entropy = dist.entropy().mean()
                loss = a_loss + 0.5*c_loss - 0.01*entropy
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


# 4. Random Agent Baseline
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    def select_action(self, state):
        return self.action_space.sample(), 0.0, 0.0


# 5. PPO Training Loop
def train_ppo(env, agent, episodes=300):
    for ep in range(1, episodes+1):
        state = env.reset()
        traj = {'states':[], 'actions':[], 'log_probs':[],
                'rewards':[], 'dones':[], 'values':[]}
        done = False
        while not done:
            a, lp, v = agent.select_action(state)
            nxt, r, done, _ = env.step(a)
            traj['states'].append(state)
            traj['actions'].append(a)
            traj['log_probs'].append(lp)
            traj['rewards'].append(r)
            traj['dones'].append(float(done))
            traj['values'].append(v)
            state = nxt
        agent.update(traj)
        if ep % 50 == 0:
            print(f"Episode {ep}/{episodes} — Reward: {sum(traj['rewards']):.2f}")


# 6. Performance Metrics
def compute_performance_metrics(equity_history, risk_free_rate=0.0):
    eq = np.array(equity_history)
    final_reward = eq[-1] - eq[0]
    roi = (final_reward / eq[0]) * 100.0
    dr = np.diff(eq) / eq[:-1]
    avg_r, std_r = dr.mean(), dr.std()
    sharpe = ((avg_r - risk_free_rate)/std_r)*np.sqrt(252) if std_r>0 else 0.0
    peak, max_dd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (peak - v)/peak
        max_dd = max(max_dd, dd)
    max_dd *= 100.0
    return roi, final_reward, sharpe, max_dd


# 7. Main: Compare Models
if __name__ == "__main__":
    path = "/Users/User/Desktop/DIA/Test/Merged_Data_For_RL/AAPL_stock_data_2022-01-01_to_2024-12-31.csv"

    tech = ["Close","MA5","MA20","RSI","MACD"]
    # Model 1: Random
    env1 = TradingEnv(path, features_to_use=tech)
    rand = RandomAgent(env1.action_space)
    s, done = env1.reset(), False
    while not done:
        a,_,_ = rand.select_action(s)
        s,_,done,_ = env1.step(a)
    eq1 = env1.equity_history
    roi1, fr1, sr1, dd1 = compute_performance_metrics(eq1)
    print(f"\nModel 1 (Random): Reward={fr1:.2f}, ROI={roi1:.2f}%, Sharpe={sr1:.2f}, MaxDD={dd1:.2f}%")

    # Model 2: PPO (tech only)
    env2 = TradingEnv(path, features_to_use=tech)
    ppo2 = PPOAgent(env2.observation_space.shape[0], env2.action_space.n)
    print("\nTraining Model 2…")
    train_ppo(env2, ppo2)
    s, done = env2.reset(), False
    while not done:
        a,_,_ = ppo2.select_action(s)
        s,_,done,_ = env2.step(a)
    eq2 = env2.equity_history
    roi2, fr2, sr2, dd2 = compute_performance_metrics(eq2)
    print(f"\nModel 2 (PPO Tech): Reward={fr2:.2f}, ROI={roi2:.2f}%, Sharpe={sr2:.2f}, MaxDD={dd2:.2f}%")

    # Model 3: PPO (tech + forecast + sentiment)
    allf = tech + ["Predicted_Next_Close","Sentiment_Score"]
    env3 = TradingEnv(path, features_to_use=allf)
    ppo3 = PPOAgent(env3.observation_space.shape[0], env3.action_space.n)
    print("\nTraining Model 3…")
    train_ppo(env3, ppo3)
    s, done = env3.reset(), False
    while not done:
        a,_,_ = ppo3.select_action(s)
        s,_,done,_ = env3.step(a)
    eq3 = env3.equity_history
    roi3, fr3, sr3, dd3 = compute_performance_metrics(eq3)
    print(f"\nModel 3 (PPO All): Reward={fr3:.2f}, ROI={roi3:.2f}%, Sharpe={sr3:.2f}, MaxDD={dd3:.2f}%")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(eq1, label="Random")
    plt.plot(eq2, label="PPO Tech")
    plt.plot(eq3, label="PPO All")
    plt.legend()
    plt.title("Equity Curves")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.show()