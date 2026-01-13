import pennylane as qml
import torch
import matplotlib.pyplot as plt
from src.modules import quantum_torch_layer

def debug_draw(test_qubits :int = 4, test_layers: int = 2) -> None:
    print(f"正在初始化 TorchLayer ({test_qubits} Qubits, {test_layers} Layers)...")
    torch_q_layer = quantum_torch_layer(test_qubits, test_layers)  # 自动创建设备、QNode 并初始化随机权重
    dummy_inputs = torch.rand(1, test_qubits)  # 准备 Dummy Input (注意：因为 interface="torch"，输入必须是 Tensor)
    target_qnode = torch_q_layer.qnode  # layer.qnode 则是底层的 PennyLane QNode
    layer_weights = torch_q_layer.weights
    print("正在绘制电路结构...")
    qml.draw_mpl(target_qnode, style='sketch')(dummy_inputs, layer_weights)
    plt.show()


if __name__ == "__main__":
    debug_draw()
