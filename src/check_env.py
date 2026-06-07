"""Small environment check for PyTorch and PennyLane."""

from __future__ import annotations

import platform
import sys

import pennylane as qml
import torch
import torchvision


def check_environment() -> None:
    """Print versions and run tiny quantum simulator tests."""
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("PennyLane:", qml.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    print("\nRunning tiny 4-qubit PennyLane default.qubit circuit...")
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for wire in range(n_qubits):
            qml.Rot(weights[wire, 0], weights[wire, 1], weights[wire, 2], wires=wire)
        for wire in range(n_qubits):
            qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    x = torch.zeros(n_qubits)
    weights = torch.zeros(n_qubits, 3, requires_grad=True)
    result = circuit(x, weights)
    print("QNode output:", result)

    print("\nRunning tiny qml.qnn.TorchLayer test...")

    @qml.qnode(dev, interface="torch")
    def layer_circuit(inputs, q_weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for wire in range(n_qubits):
            qml.Rot(q_weights[wire, 0], q_weights[wire, 1], q_weights[wire, 2], wires=wire)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    qlayer = qml.qnn.TorchLayer(layer_circuit, {"q_weights": (n_qubits, 3)})
    y = qlayer(torch.zeros(n_qubits))
    print("TorchLayer output shape:", tuple(y.shape))

    if y.shape[-1] != n_qubits:
        raise RuntimeError("TorchLayer returned an unexpected shape.")

    print("\nEnvironment check passed")


if __name__ == "__main__":
    check_environment()

