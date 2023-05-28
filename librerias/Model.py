import torch as torch
import numpy as np


class DQN(torch.nn.Module):

    def __init__(self, env, net, learning_rate=1e-3, device='cpu'):
        super(DQN, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        ### Construcción de la red neuronal
        self.model = net

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        ### Se ofrece la opción de trabajar con CUDA
        if self.device == 'cuda':
            self.model.cuda()

    ### Método e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)  # acción aleatoria
        else:
            qvals = self.get_qvals(state)  # acción a partir del cálculo del valor de Q para esa acción
            action = torch.max(qvals, dim=-1)[1].item()
        return action

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.model(state_t)


class PGReinforce(torch.nn.Module):

    def __init__(self, env, net, learning_rate=1e-3, device='cpu'):
        super(PGReinforce, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate

        ### Construcción de la red neuronal
        self.model = torch.nn.Sequential(net, torch.nn.Softmax(dim=-1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        ### Se ofrece la opción de trabajar con cuda
        if self.device == 'cuda':
            self.model.cuda()

    # Obtención de las probabilidades de las posibles acciones
    def get_action_prob(self, state):
        action_probs = self.model(torch.FloatTensor(state))
        return action_probs