import pennylane as qml


def _circuit_ansatz(inputs, weights, n_qubits):
    """
    定义具体的量子门操作。
    """
    for i in range(n_qubits):
        qml.RY(inputs[:,i], wires=i)
        qml.Hadamard(wires=i+n_qubits)
    for j, layer_weights in enumerate(weights):
        for i in range(n_qubits):
            qml.CRot(layer_weights[i][0], layer_weights[i][1], layer_weights[i][2], wires=[i+n_qubits, i])
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def quantum_torch_layer(n_qubits, n_layers, q_device_name="default.qubit"):
    """
    负责初始化设备、QNode，并打包成一个 PyTorch 兼容的 Layer 返回。
    """
    dev = qml.device(q_device_name, wires=2 * n_qubits)  # 初始化设备
    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        return _circuit_ansatz(inputs, weights, n_qubits)
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}  # 定义权重形状 (PennyLane 要求)
    return qml.qnn.TorchLayer(qnode, weight_shapes)  # 返回准备好的 TorchLayer
