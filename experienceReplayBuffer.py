from collections import namedtuple, deque
import numpy as np

class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = namedtuple('Buffer',
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    ##Creamos una lista de índices aleatorios y empaquetamos las experiencias en arrays de Numpy (facilita el cálculo posterior de la pérdida)
    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    ## Se añaden las nuevas experiencias
    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.buffer(state, action, reward, done, next_state))

    ## Rellenamos el buffer con experiencias aleatorias al inicio del entrenamiento
    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in