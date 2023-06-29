import torch as torch
import numpy as np


class DQN(torch.nn.Module):

    def __init__(self, env, net, learning_rate=1e-3, device='cpu', classic=False, lr_input=0.01, lr_output=0.01):
        super(DQN, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self.lr_input = lr_input
        self.lr_output = lr_output

        ### Construcción de la red
        self.model = net

        if classic == True:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

        else:
            self.params = []
            self.params.append({'params': self.model.q_layers.parameters()})
#            if hasattr(self.model, 'w_input') and self.model.w_input is not None:
            self.params.append({'params': self.model.w_input, 'lr': self.lr_input})
#            if hasattr(self.model, 'w_output') and self.model.w_output is not None:
            self.params.append({'params': self.model.w_output, 'lr': self.lr_output})
            self.optimizer = torch.optim.RMSprop(self.params, lr=self.learning_rate)

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

    def __init__(self, env, net, learning_rate=1e-3, device='cpu', classic=False, lr_input=0.01, lr_output=0.01):
        super(PGReinforce, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate
        self.lr_input = lr_input
        self.lr_output = lr_output


        if classic == True:
            self.model = torch.nn.Sequential(net, torch.nn.Softmax(dim=-1))
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

        else:
            self.model = net
            self.params = []
            self.params.append({'params': self.model.q_layers.parameters()})
            self.params.append({'params': self.model.w_input, 'lr': self.lr_input})
            self.params.append({'params': self.model.w_output, 'lr': self.lr_output})
            self.model = torch.nn.Sequential(net, torch.nn.Softmax(dim=-1))
            self.optimizer = torch.optim.RMSprop(self.params, lr=self.learning_rate)


        ### Se ofrece la opción de trabajar con cuda
        if self.device == 'cuda':
            self.model.cuda()

    # Obtención de las probabilidades de las posibles acciones
    def get_action_prob(self, state):
        action_probs = self.model(torch.FloatTensor(state))
        return action_probs