from copy import deepcopy, copy
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

class DQNAgent:

    def __init__(self, env, dnnetwork, target_network, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32):

        self.env = env
        self.dnnetwork = dnnetwork
        #        self.target_network = deepcopy(dnnetwork) # red objetivo (copia de la principal)
        self.target_network = target_network  # red objetivo (copia de la principal)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100  # bloque de los X últimos episodios de los que se calculará la media de recompensa
        self.reward_threshold = self.env.spec.reward_threshold  # recompensa media a partir de la cual se considera
        # que el agente ha aprendido a jugar
        self.initialize()

    def initialize(self):
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = self.env.reset()

    ## Tomamos una nueva acción
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # acción aleatoria en el burn-in y en la fase de exploración (epsilon)
            action = self.env.action_space.sample()
        else:
            # acción a partir del valor de Q (elección de la acción con mejor Q)
            action = self.dnnetwork.get_action(self.state0, eps)
            self.step_count += 1

        # Realizamos la acción y obtenemos el nuevo estado y la recompensa
        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state)  # guardamos experiencia en el buffer
        self.state0 = new_state.copy()

        if done:
            self.state0 = self.env.reset()
        return done

    ## Entrenamiento
    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000):

        self.gamma = gamma

        # Rellenamos el buffer con N experiencias aleatorias ()
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Training...")
        while training:
            self.state0 = self.env.reset()
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                # El agente toma una acción

                gamedone = self.take_step(self.epsilon, mode='train')

                # Actualizamos la red principal según la frecuencia establecida
                if self.step_count % dnn_update_frequency == 0:
                    self.update()

                # Sincronizamos la red principal y la red objetivo según la frecuencia establecida
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.dnnetwork.state_dict())
                    self.sync_eps.append(episode)

                if gamedone:
                    episode += 1
                    self.training_rewards.append(self.total_reward)  # guardamos las recompensas obtenidas
                    self.update_loss = []
                    mean_rewards = np.mean(  # calculamos la media de recompensa de los últimos X episodios
                        self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)

                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    # Comprobamos que todavía quedan episodios
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break

                    # Termina el juego si la media de recompensas ha llegado al umbral fijado para este juego
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        break

                    # Actualizamos epsilon según la velocidad de decaimiento fijada
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

    ## Cálculo de la pérdida
    def calculate_loss(self, batch):
        # Separamos las variables de la experiencia y las convertimos a tensores
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.dnnetwork.device)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(
            device=self.dnnetwork.device)
        dones_t = torch.ByteTensor(dones).to(device=self.dnnetwork.device)

        # Obtenemos los valores de Q de la red principal
        qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions_vals)

        # Obtenemos los valores de Q objetivo. El parámetro detach() evita que estos valores actualicen la red objetivo
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()

        qvals_next[dones_t] = 0  # 0 en estados terminales

        # Calculamos la ecuación de Bellman
        expected_qvals = self.gamma * qvals_next + rewards_vals

        # Calculamos la pérdida
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1, 1))

        return loss

    def update(self):

        self.dnnetwork.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado

        batch = self.buffer.sample_batch(batch_size=self.batch_size)  # seleccionamos un conjunto del buffer

        loss = self.calculate_loss(batch)  # calculamos la pérdida

        loss.backward()  # hacemos la diferencia para obtener los gradientes

        self.dnnetwork.optimizer.step()  # aplicamos los gradientes a la red neuronal

        # Guardamos los valores de pérdida
        if self.dnnetwork.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def plot_rewards(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_rewards, label='Rewards')
        plt.plot(self.mean_training_rewards, label='Mean Rewards')
        plt.axhline(self.reward_threshold, color='r', label="Reward threshold")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        plt.show()


class reinforceAgent:

    def __init__(self, env, pgnetwork):

        self.env = env
        self.pgnetwork = pgnetwork
        self.nblock = 100  # bloque de los X últimos episodios de los que se calculará la media de recompensa
        self.reward_threshold = self.env.spec.reward_threshold  # recompensa media a partir de la cual se considera
        # que el agente ha aprendido a jugar
        self.initialize()

    def initialize(self):
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1
        self.training_rewards = []
        self.mean_training_rewards = []
        self.update_loss = []

    ## Entrenamiento
    def train(self, gamma=0.99, max_episodes=2000, batch_size=10):
        self.gamma = gamma
        self.batch_size = batch_size

        episode = 0
        action_space = np.arange(self.env.action_space.n)
        training = True
        print("Training...")
        while training:
            state0 = self.env.reset()
            episode_states = []
            episode_rewards = []
            episode_actions = []
            gamedone = False

            while gamedone == False:
                # Obtenemos las acciones
                action_probs = self.pgnetwork.get_action_prob(state0).detach().numpy()
                action = np.random.choice(action_space, p=action_probs)
                next_state, reward, gamedone, _ = self.env.step(action)

                # Almacenamos las experiencias que se van obteniendo en este episodio
                episode_states.append(state0)
                episode_rewards.append(reward)
                episode_actions.append(action)
                state0 = next_state

                if gamedone:
                    episode += 1
                    # Calculamos el término del retorno menos la línea de base
                    self.batch_rewards.extend(self.discount_rewards(episode_rewards))
                    self.batch_states.extend(episode_states)
                    self.batch_actions.extend(episode_actions)
                    self.training_rewards.append(sum(episode_rewards))  # guardamos las recompensas obtenidas

                    # Actualizamos la red cuando se completa el tamaño del batch
                    if self.batch_counter == self.batch_size:
                        self.update(self.batch_states, self.batch_rewards, self.batch_actions)
                        self.update_loss = []

                        # Reseteamos las variables del epsiodio
                        self.batch_rewards = []
                        self.batch_actions = []
                        self.batch_states = []
                        self.batch_counter = 1

                    # Actualizamos el contador del batch
                    self.batch_counter += 1

                    # Calculamos la media de recompensa de los últimos X episodios
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)

                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        episode, mean_rewards), end="")

                    # Comprobamos que todavía quedan episodios
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break

                    # Termina el juego si la media de recompensas ha llegado al umbral fijado para este juego
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        break

    def discount_rewards(self, rewards):
        discount_r = np.zeros_like(rewards)
        timesteps = range(len(rewards))
        reward_sum = 0
        for i in reversed(timesteps):  # revertimos la dirección del vector para hacer la suma cumulativa
            reward_sum = rewards[i] + self.gamma * reward_sum
            discount_r[i] = reward_sum
        baseline = np.mean(discount_r)  # establecemos la media de la recompensa como línea de base
        return discount_r - baseline

        ## Actualización

    def update(self, batch_s, batch_r, batch_a):
        self.pgnetwork.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado
        state_t = torch.FloatTensor(batch_s)
        reward_t = torch.FloatTensor(batch_r)
        action_t = torch.LongTensor(batch_a)
        loss = self.calculate_loss(state_t, action_t, reward_t)  # calculamos la pérdida
        loss.backward()  # hacemos la diferencia para obtener los gradientes
        self.pgnetwork.optimizer.step()  # aplicamos los gradientes a la red neuronal
        # Guardamos los valores de pérdida
        if self.pgnetwork.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    ## Cálculo de la pérdida
    # Recordatorio: cada actualización es proporcional al producto del retorno y el gradiente de la probabilidad
    # de tomar la acción tomada, dividido por la probabilidad de tomar esa acción (logaritmo natural)
    def calculate_loss(self, state_t, action_t, reward_t):
        logprob = torch.log(self.pgnetwork.get_action_prob(state_t))
        selected_logprobs = reward_t * \
                            logprob[np.arange(len(action_t)), action_t]
        loss = -selected_logprobs.mean()
        return loss

    def plot_rewards(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_rewards, label='Rewards')
        plt.plot(self.mean_training_rewards, label='Mean Rewards')
        plt.axhline(self.reward_threshold, color='r', label="Reward threshold")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        plt.show()