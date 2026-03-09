import os

import pennylane as qml
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService, Session

# ================= 配置区域 (Configuration) =================
load_dotenv()
API_KEY = os.getenv("IBM_API_KEY")
CRN_STRING = os.getenv("IBM_CRN_STRING")
TARGET_BACKEND = "ibm_kobe"
SHOTS = 4096
# ==========================================================


def get_ibm_device(n_qubits_logic):
    """
    连接 IBM 并在指定后端上创建 PennyLane 设备。
    """
    required_wires = n_qubits_logic * 2

    print("[IBM] Connecting to service...")
    try:
        service = QiskitRuntimeService(channel="ibm_cloud", token=API_KEY, instance=CRN_STRING)
    except Exception as e:
        print(f"[Error] Login failed: {e}")
        return None

    print(f"[IBM] Getting backend: {TARGET_BACKEND}...")
    try:
        backend = service.backend(TARGET_BACKEND)
        print(f"[IBM] Backend found: {backend.name} (Total Qubits: {backend.num_qubits})")

        if backend.num_qubits < required_wires:
            raise ValueError(f"设备比特数不足！需要 {required_wires}，但 {backend.name} 只有 {backend.num_qubits}。")

    except Exception as e:
        print(f"[Error] Backend error: {e}")
        return None

    print(f"[IBM] Creating PennyLane device with {required_wires} wires...")
    # 使用 qiskit.remote 连接真实硬件
    try:
        ibm_session = Session(backend=backend, max_time="2h")
    except Exception as e:
        print(f"[Warning] Could not create Session: {e}. Falling back to standard mode.")
        ibm_session = None

    try:
        # 3. 将 session 对象传给 device
        dev = qml.device(
            "qiskit.remote",
            wires=required_wires,
            backend=backend,
            shots=SHOTS,
            session=ibm_session,  # <--- 传对象，不要传 True
        )
        return dev
    except Exception as e:
        print(f"[Error] Device creation failed: {e}")
        # 如果 Session 创建失败，尝试回退到无 Session 模式
        # 虽然慢一点，但至少能跑
        print("[System] Retrying without Session...")
        return qml.device("qiskit.remote", wires=required_wires, backend=backend, shots=SHOTS)
