
import pennylane as qml
import torch as torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)

def layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])

def measure_cartpole(n_qubits):
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
    ]


def measure_acrobot(n_qubits):
    return [
        qml.expval(qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2)),
        qml.expval(qml.PauliZ(2)@qml.PauliZ(3)@qml.PauliZ(5)),
        qml.expval(qml.PauliZ(4)@qml.PauliZ(5)@qml.PauliZ(0)@qml.PauliZ(1))
    ]

def measure_acrobot(n_qubits):
    return [
        qml.expval(qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(4)),
        qml.expval(qml.PauliZ(2)@qml.PauliZ(3)@qml.PauliZ(5)),
        qml.expval(qml.PauliZ(4)@qml.PauliZ(5)@qml.PauliZ(0)@qml.PauliZ(1))
    ]

def measure_mountaincar(n_qubits):
    return [
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(0)@qml.PauliZ(1))
    ]

def get_model(n_qubits, n_layers, environment, quantum_device):
    dev = qml.device(quantum_device, wires=n_qubits)
    shapes = {
        "y_weights": (n_layers, n_qubits),
        "z_weights": (n_layers, n_qubits)
    }

    @qml.qnode(dev, interface='torch')
    def circuit_cartpole(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure_cartpole(n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit_acrobot(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure_acrobot(n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit_mountaincar(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure_mountaincar(n_qubits)

    if environment == 0:  # Cartpole
        model = qml.qnn.TorchLayer(circuit_cartpole, shapes)
    else: # Acrobot
        if environment == 1:  # Acrobot
            model = qml.qnn.TorchLayer(circuit_acrobot, shapes)
        else: # Mountaincar
            model = qml.qnn.TorchLayer(circuit_mountaincar, shapes)
    return model

class QuantumNet(nn.Module):
    def __init__(self, n_layers, n_qubits, n_actions, environment, quantum_device):
        super(QuantumNet, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.environment = environment
#        if self.environment == 0:  # Cartpole
        self.w_input = Parameter(torch.Tensor(self.n_qubits))
        nn.init.normal_(self.w_input, mean=0.)
        self.w_output = Parameter(torch.Tensor(self.n_actions))
        nn.init.normal_(self.w_output, mean=90.)
        self.q_layers = get_model(n_qubits=self.n_qubits,
                                  n_layers=n_layers,
                                  environment=self.environment, quantum_device=quantum_device)

    def forward(self, inputs):
        if self.environment == 0:  # Cartpole
            inputs = torch.atan(inputs)

        else:  # Acrobot
            inputs[0] = inputs[0] * torch.pi/2
            inputs[1] = inputs[1] * torch.pi/2
            inputs[2] = inputs[2] * torch.pi/2
            inputs[3] = inputs[3] * torch.pi/2
            inputs[4] = torch.atan(inputs[4])
            inputs[5] = torch.atan(inputs[5])

        inputs = inputs * self.w_input

        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2

        outputs = outputs * self.w_output
        return outputs


