import timeit
#%%

import torch
#%%
from librerias import Agent, Model, QVC, experienceReplayBuffer as erb
#%%
import gym as gym

from codecarbon import EmissionsTracker

#%%
#%%
env = gym.envs.make("CartPole-v0")
#%%
# Comprobación de la versión de GYM instalada
print('La versión de gym instala: ' + gym.__version__)
# Comprobación de entorno con gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("El entorno utiliza: ", device)
#%%
#Visualizamos el entorno
for i_episode in range(15):
    observation = env.reset()
    for t in range(100):
        #env.render() #EL RENDER SÓLO FUNCIONA EN LOCAL: comentar línea si no se está en local.
        action = env.action_space.sample() #acción aleatoria
        observation, reward, done, info = env.step(action) #ejecución de la acción elegida
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close() #cerramos la visualización del entorno
#%%
lr = 0.001            #Velocidad de aprendizaje
MEMORY_SIZE = 100000  #Máxima capacidad del buffer
MAX_EPISODES = 5   #Número máximo de episodios (el agente debe aprender antes de llegar a este valor)
EPSILON = 1           #Valor inicial de epsilon
EPSILON_DECAY = .99   #Decaimiento de epsilon
GAMMA = 0.99          #Valor gamma de la ecuación de Bellman
BATCH_SIZE = 32       #Conjunto a coger del buffer para la red neuronal
BURN_IN = 1000        #Número de episodios iniciales usados para rellenar el buffer antes de entrenar
DNN_UPD = 10           #Frecuencia de actualización de la red neuronal
DNN_SYNC = 30       #Frecuencia de sincronización de pesos entre la red neuronal y la red objetivo
#%%
n_layers = 5
#%%
buffer = erb.experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
#%%
n_qubits = 4
n_actions = 2
environment = 0 # Cartpole
quantum_device = "default.qubit"
#%%
# Networks
net = QVC.QuantumNet(n_layers, n_qubits, n_actions, environment, quantum_device)
#%%
target_network = QVC.QuantumNet(n_layers, n_qubits, n_actions, environment, quantum_device)
#%%
dqn = Model.DQN(env, net, learning_rate=lr)
#%%
dqn_target = Model.DQN(env, target_network, learning_rate=lr)
#%%
agent = Agent.DQNAgent(env, dqn, dqn_target, buffer, EPSILON, EPSILON_DECAY, BATCH_SIZE)
#%%
tracker = EmissionsTracker()
tracker.start()
tiempo_inicio = timeit.default_timer()
#%%


agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES,
              batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)
#%%



tiempo_ejecucion = round(timeit.default_timer() - tiempo_inicio, 0)
print("Tiempo ejecución entrenamiento: " + str(int(tiempo_ejecucion/3600))
      + " horas, " + str(int((tiempo_ejecucion % 3600)/60)) + " minutos y "
      + str(int((tiempo_ejecucion % 3600)%60)) + " segundos")

emissions: float = tracker.stop()
print(emissions)

#%%
agent.plot_rewards()