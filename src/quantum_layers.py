"""PennyLane quantum layer for the simulated hybrid model.

This project uses the PennyLane ``default.qubit`` simulator only. It does not require
access to real quantum hardware and does not claim quantum advantage.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


class PennyLaneQuantumLayer(nn.Module):
    """A small variational quantum circuit wrapped as a PyTorch module.

    Circuit design:
    inputs -> AngleEmbedding -> Rot gates -> CNOT ring -> PauliZ measurements
    """

    def __init__(self, n_qubits: int = 4, n_quantum_layers: int = 2):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_quantum_layers = int(n_quantum_layers)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            for layer_idx in range(self.n_quantum_layers):
                for wire in range(self.n_qubits):
                    qml.Rot(
                        weights[layer_idx, wire, 0],
                        weights[layer_idx, wire, 1],
                        weights[layer_idx, wire, 2],
                        wires=wire,
                    )
                for wire in range(self.n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]

        weight_shapes = {"weights": (self.n_quantum_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the quantum layer.

        The explicit loop over a batch is slower but beginner-friendly and reliable
        across PennyLane versions.
        """
        if inputs.ndim == 1:
            return self.qlayer(inputs)
        outputs = [self.qlayer(sample) for sample in inputs]
        return torch.stack(outputs, dim=0)

