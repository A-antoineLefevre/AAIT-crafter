import argparse
import pickle
from pathlib import Path
import random
from collections import deque # Ajout nécessaire pour ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.crafter_wrapper import Env


class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()

# NOUVEAU: Implémentation du Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir et stacker les Tensors PyTorch
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1) 
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# NOUVEAU: Implémentation du Réseau Q (QNetwork)
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # CORRECTION: Utilisation de Conv2d
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)


    def forward(self, x):
        # x: (Batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# NOUVEAU: Implémentation de l'Agent DQN
class DQNAgent:
    # CORRECTION: Ajout des arguments de fréquence et d'exploration
    def __init__(self, action_num, device, history_length, 
                 gamma=0.99, lr=1e-4,
                 target_update_frequency=10000, 
                 learning_start_steps=50000, 
                 learning_frequency=4,
                 epsilon_start=1.0, 
                 epsilon_final=0.01,
                 epsilon_decay_steps=250000):

        self.action_num = action_num
        self.device = device
        self.gamma = gamma
        
        # Hyperparamètres d'exploration
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start # Taux d'exploration actuel

        # Hyperparamètres de fréquence
        self.target_update_frequency = target_update_frequency
        self.learning_start_steps = learning_start_steps
        self.learning_frequency = learning_frequency

        self.q_net = QNetwork(history_length, action_num).to(device)
        self.target_net = QNetwork(history_length, action_num).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict()) 
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=20000)

    # CORRECTION: Ajout de step_cnt pour la décroissance d'epsilon
    def act(self, observation, step_cnt):
        # Décroissance linéaire de epsilon
        self.epsilon = max(
            self.epsilon_final, 
            self.epsilon_start - (self.epsilon_start - self.epsilon_final) * min(1.0, step_cnt / self.epsilon_decay_steps)
        )
        
        if random.random() < self.epsilon:
            # Exploration
            return random.randrange(self.action_num)
        else:
            # Exploitation
            with torch.no_grad(): 
                q_values = self.q_net(observation.unsqueeze(0)) 
                return q_values.argmax().item()
            
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # 1. Échantillonner le mini-lot
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, self.device)

        # 2. Calculer les Q-valeurs actuelles Q(s, a)
        q_current = self.q_net(states).gather(1, actions.long())

        # 3. Calculer les Q-cibles Q_target (DOUBLE DQN)
        with torch.no_grad():
            # SÉLECTION (Selection) : Utiliser le réseau Q actuel (q_net) pour trouver la MEILLEURE action (a*) dans le prochain état s'.
            a_prime = self.q_net(next_states).argmax(1).unsqueeze(1)
            
            # ÉVALUATION (Evaluation) : Utiliser le réseau CIBLE (target_net) pour évaluer Q(s', a*).
            q_next_max = self.target_net(next_states).gather(1, a_prime)
            
            # Calculer la cible finale de Bellman
            q_target = rewards + self.gamma * q_next_max * (1 - dones.float())

        # 4. Calculer la perte et optimiser
        loss = self.loss_fn(q_current, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    # NOTE: Pour l'évaluation, le taux d'exploration doit être proche de 0 (exploitation pure).
    # Puisque l'agent.act() utilise self.epsilon (qui est en décroissance), c'est acceptable, 
    # mais il faudrait idéalement forcer agent.epsilon = 0.0 pour l'évaluation.

    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            # Passe step_cnt=opt.steps à act pour utiliser epsilon_final (exploitation)
            # ou vous pourriez définir une fonction act_eval dans l'agent.
            action = agent.act(obs, opt.steps) 
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    #opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    
    # PASSAGE DES ARGUMENTS À DQNAgent
    agent = DQNAgent(
        env.action_space.n, 
        opt.device, 
        opt.history_length, 
        target_update_frequency=opt.target_update_frequency,
        learning_start_steps=opt.learning_start_steps,
        learning_frequency=opt.learning_frequency,
        epsilon_start=opt.epsilon_start,
        epsilon_final=opt.epsilon_final,
        epsilon_decay_steps=opt.epsilon_decay_steps
    )
    
    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    
    while step_cnt < opt.steps or not done:
        
        # LOGIQUE D'OBSERVATION PRÉCÉDENTE POUR LE BUFFER
        if not done:
            prev_obs = obs
            
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False
            prev_obs = obs # Définit prev_obs pour la première étape de l'épisode

        # PASSAGE DE step_cnt À act POUR LA DÉCROISSANCE D'EPSILON
        action = agent.act(obs, step_cnt)
        
        obs, reward, done, info = env.step(action)
        
        # Stockage de la transition (s, a, r, s', done)
        agent.store_transition(prev_obs, action, reward, obs, done)

        step_cnt += 1

        # Logique d'apprentissage DQN
        if step_cnt > opt.learning_start_steps and step_cnt % opt.learning_frequency == 0:
            agent.learn(batch_size=32)

        # Logique de mise à jour du réseau cible
        if step_cnt % opt.target_update_frequency == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    # ARGUMENTS D'APPRENTISSAGE DQN
    parser.add_argument(
        "--learning-start-steps",
        type=int,
        default=50000, 
        help="Number of steps before learning starts.",
    )
    parser.add_argument(
        "--learning-frequency",
        type=int,
        default=4, 
        help="Number of steps between learning updates.",
    )
    parser.add_argument(
        "--target-update-frequency",
        type=int,
        default=10000, 
        help="Number of steps between target network updates.",
    )
    # ARGUMENTS D'EXPLORATION EPSILON
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial value of epsilon for exploration.",
    )
    parser.add_argument(
        "--epsilon-final",
        type=float,
        default=0.01,
        help="Final value of epsilon after decay.",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=250000,
        help="Number of steps over which epsilon decays.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())