import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from concurrent.futures import ProcessPoolExecutor
from qiskit_aer import AerSimulator
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, transpile
import ast
import itertools
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


def makeError(phi: float, disp: float) -> float:
    random_number = np.random.normal(0, (disp * disp), None)
    # print(random_number)
    return phi + random_number


def QFT(circuit: QuantumCircuit, start: int, n: int, disp: float = 0) -> QuantumCircuit:
    for i in range(n - 1, start - 1, -1):
        circuit.h(i)
        for j in range(i - 1, start - 1, -1):
            phase = np.pi / pow(2, i - j)
            phase = makeError(phase, disp)
            circuit.cp(phase, j, i)
    return circuit


def IQFT(
    circuit: QuantumCircuit, start: int, n: int, disp: float = 0
) -> QuantumCircuit:
    for i in range(start, n, 1):
        for j in range(start, i, 1):
            phase = -np.pi / pow(2, i - j)
            phase = makeError(phase, disp)
            circuit.cp(phase, j, i)
        circuit.h(i)
    return circuit


def copy_qubit(
    circuit: QuantumCircuit, qubit: int, an1: int, an2: int
) -> QuantumCircuit:

    circuit.cx(qubit, an1)
    circuit.cx(qubit, an2)
    return circuit


def qubit_flip_correction(
    circuit: QuantumCircuit, qubit: int, an1: int, an2: int
) -> QuantumCircuit:
    circuit.cx(qubit, an1)
    circuit.cx(qubit, an2)

    circuit.ccx(an2, an1, qubit)
    return circuit


def QFT_adder_const(
    circuit: QuantumCircuit,
    const: int,
    b_start: int,
    n: int,
    inverse: bool = False,
    disp: float = 0,
) -> QuantumCircuit:
    sign = 1 if not inverse else -1
    for j in range(n):
        target_qbit = b_start + j
        phase = sign * np.pi * const / (1 << j)
        phase = makeError(phase, disp)
        circuit.p(phase, target_qbit)
    return circuit


def QFT_adder_const_with_correction(
    circuit: QuantumCircuit,
    const: int,
    b_start: int,
    n: int,
    inverse: bool,
    disp: float = 0,
) -> QuantumCircuit:
    new_qubits_count = 2 * circuit.num_qubits
    new_qr = QuantumRegister(new_qubits_count, "ancilla")
    circuit.add_register(new_qr)
    circuit.barrier(label="copy")
    for i in range(0, n):
        circuit = copy_qubit(circuit, b_start + i, b_start + i + n, b_start + i + 2 * n)

    for i in range(3):
        circuit.barrier(label="qft")
        circuit = QFT(circuit, i * n, i * n + n, disp)
        circuit.barrier(label="sum")
        circuit = QFT_adder_const(circuit, const, i * n, n, inverse, disp)
        circuit.barrier(label="iqft")
        circuit = IQFT(circuit, i * n, i * n + n, disp)

    circuit.barrier(label="correction")
    for i in range(0, n):
        circuit = qubit_flip_correction(
            circuit, b_start + i, b_start + i + n, b_start + i + 2 * n
        )
    return circuit


def generate_all_inputs(filename="dataset2.txt", num_random=100):
    n_states = 8
    n_const = 8
    amp = round(float(1 / np.sqrt(2)), 5)

    records = []

    # 1. Базисные векторы (64 записи)
    for i in range(n_states):
        vec = [0] * n_states
        vec[i] = 1
        for c in range(n_const):
            records.append(f"{vec} {c}\n")

    # 2. Суперпозиции (224 записи)
    pairs = list(itertools.combinations(range(n_states), 2))
    for p1, p2 in pairs:
        vec = [0.0] * n_states
        vec[p1] = amp
        vec[p2] = amp
        clean_vec = [float(x) for x in vec]
        for c in range(n_const):
            records.append(f"{clean_vec} {c}\n")

    # 3. СЛУЧАЙНЫЕ ВЕКТОРЫ (num_random * 8 констант)
    print(f"Генерация {num_random} случайных состояний...")
    for _ in range(num_random):
        # Генерируем случайные комплексные амплитуды (если нужны только вещественные - убираем j)
        # Для простоты нейронки обычно используют вещественные векторы:
        raw_vec = np.random.randn(n_states)

        # Нормализуем: сумма квадратов должна быть равна 1
        norm = np.linalg.norm(raw_vec)
        normalized_vec = raw_vec / norm

        # Превращаем в чистый список Python с округлением
        clean_vec = [round(float(x), 5) for x in normalized_vec]

        for c in range(n_const):
            records.append(f"{clean_vec} {c}\n")

    with open(filename, "w") as f:
        f.writelines(records)


def process_with_simulator(filename="2-dataset.txt"):
    updated_rows = []
    backend1 = AerSimulator()
    backend2 = FakeSherbrooke()

    def to_clean_list(raw_vec):
        clean = []
        for x in raw_vec:
            val = float(x.real)  # Избавляемся от комплексности и numpy-типов
            if abs(val) < 1e-10:
                clean.append(0)  # Убираем шум и -0.0
            elif val.is_integer():
                clean.append(int(val))  # 1.0 -> 1
            else:
                clean.append(val)
        return clean

    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    for line in lines:
        if not line.strip():
            continue

        clean_line = line.replace("np.float64(", "").replace(")", "")
        parts = clean_line.strip().rsplit(" ", 1)

        try:
            input_vec_raw = ast.literal_eval(parts[0])
            const = int(parts[1])
        except Exception as e:
            continue

        n_qubits = 3
        q = QuantumCircuit(n_qubits)
        q.initialize(input_vec_raw, range(n_qubits), normalize=True)

        q = QFT(q, 0, n_qubits)
        q = QFT_adder_const(q, const, 0, n_qubits)
        q = IQFT(q, 0, n_qubits)

        q.measure_all()
        job1 = backend1.run(transpile(q, backend1), shots=1000)
        job2 = backend2.run(transpile(q, backend2), shots=1000)
        counts1 = job1.result().get_counts()
        counts2 = job2.result().get_counts()

        raw_res1 = [
            counts1.get(format(i, f"0{n_qubits}b"), 0) / 1000
            for i in range(2**n_qubits)
        ]
        raw_res2 = [
            counts2.get(format(i, f"0{n_qubits}b"), 0) / 1000
            for i in range(2**n_qubits)
        ]

        # 5. Очистка векторов перед записью (убираем np.float64 и лишний шум)
        input_vec = to_clean_list(input_vec_raw)
        output_vec1 = to_clean_list(raw_res1)
        output_vec2 = to_clean_list(raw_res2)

        # 6. Запись в файл: [вход] константа [идеал_100_шотов] [шумный_100_шотов]
        updated_rows.append(f"{input_vec} {const} {output_vec1} {output_vec2}\n")
        i += 1
        print(i)
    with open(filename, "w") as f:
        f.writelines(updated_rows)


if __name__ == "__main__":
    generate_all_inputs()  # Создаем список задач
    process_with_simulator()  # Решаем их на квантовом симуляторе
