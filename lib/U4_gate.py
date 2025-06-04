'''
定义了量子比特数为4的量子电路。

每个特定的量子逻辑门被抽象为一个类，包含了量子逻辑门的大小、参数数量和电路结构。
'''

import pennylane as qml

'''
CB结构的U4门
'''
# RY + CB结构CRZ门
class u4_cb_1:
    def __init__(self):
        self.gate_size=4
        self.weight_size = 8

    def circuit(self, weights, wires):
        assert(len(wires) >= 4)
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RY(weights[2], wires=wires[2])
        qml.RY(weights[3], wires=wires[3])

        qml.CRZ(weights[4], wires=[wires[3], wires[0]])
        qml.CRZ(weights[5], wires=[wires[2], wires[3]])
        qml.CRZ(weights[6], wires=[wires[1], wires[2]])
        qml.CRZ(weights[7], wires=[wires[0], wires[1]])

# RX + CB结构CRY门
class u4_cb_2:
    def __init__(self):
        self.gate_size=4
        self.weight_size = 8

    def circuit(self, weights, wires):
        assert(len(wires) >= 4)
        qml.RX(weights[0], wires=wires[0])
        qml.RX(weights[1], wires=wires[1])
        qml.RX(weights[2], wires=wires[2])
        qml.RX(weights[3], wires=wires[3])

        qml.CRY(weights[4], wires=[wires[3], wires[0]])
        qml.CRY(weights[5], wires=[wires[2], wires[3]])
        qml.CRY(weights[6], wires=[wires[1], wires[2]])
        qml.CRY(weights[7], wires=[wires[0], wires[1]])
    
#吧RY&RZ + NN结构CZ门 + RY&RZ + CZ
class u4_nn_1:
    def __init__(self):
        self.gate_size=4
        self.weight_size = 12

    # circuit 12
    def circuit(self, weights, wires):
        assert(len(wires) >= 4)
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RY(weights[2], wires=wires[2])
        qml.RY(weights[3], wires=wires[3])
        qml.RZ(weights[4], wires=wires[0])
        qml.RZ(weights[5], wires=wires[1])
        qml.RZ(weights[6], wires=wires[2])
        qml.RZ(weights[7], wires=wires[3])

        # NN结构的CZ门
        qml.CZ(wires=[wires[0], wires[2]])
        qml.CZ(wires=[wires[1], wires[3]])

        qml.RY(weights[8], wires=wires[1])
        qml.RY(weights[9], wires=wires[2])
        qml.RZ(weights[10], wires=wires[1])
        qml.RZ(weights[11], wires=wires[2])

        qml.CZ(wires=[wires[1], wires[2]])

# RY&RZ + NN结构CNOT门 + RY&RZ + CZ
class u4_nn_2:
    def __init__(self):
        self.gate_size=4
        self.weight_size = 12

    def circuit(self, weights, wires):
        assert(len(wires) >= 4)
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RY(weights[2], wires=wires[2])
        qml.RY(weights[3], wires=wires[3])
        qml.RZ(weights[4], wires=wires[0])
        qml.RZ(weights[5], wires=wires[1])
        qml.RZ(weights[6], wires=wires[2])
        qml.RZ(weights[7], wires=wires[3])

        # NN结构的CZ门
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.CNOT(wires=[wires[1], wires[3]])

        qml.RY(weights[8], wires=wires[1])
        qml.RY(weights[9], wires=wires[2])
        qml.RZ(weights[10], wires=wires[1])
        qml.RZ(weights[11], wires=wires[2])

        qml.CNOT(wires=[wires[1], wires[2]])

# RY&RZ + AA结构的CRY门 + RY&RZ
class u4_aa_1:
    def __init__(self):
        self.gate_size = 4
        self.weight_size = 20
    
    def circuit(self, weights, wires):
        assert(len(wires) >= 4)
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RY(weights[2], wires=wires[2])
        qml.RY(weights[3], wires=wires[3])
        qml.RZ(weights[4], wires=wires[0])
        qml.RZ(weights[5], wires=wires[1])
        qml.RZ(weights[6], wires=wires[2])
        qml.RZ(weights[7], wires=wires[3])
        
        qml.CRY(weights[8], wires=[wires[0], wires[3]])
        qml.CRY(weights[9], wires=[wires[0], wires[2]])
        qml.CRY(weights[10], wires=[wires[0], wires[1]])
        qml.CRY(weights[11], wires=[wires[1], wires[0]])
        qml.CRY(weights[12], wires=[wires[1], wires[2]])
        qml.CRY(weights[13], wires=[wires[1], wires[3]])
        qml.CRY(weights[14], wires=[wires[2], wires[0]])
        qml.CRY(weights[15], wires=[wires[2], wires[1]])
        qml.CRY(weights[16], wires=[wires[2], wires[3]])
        qml.CRY(weights[17], wires=[wires[3], wires[0]])
        qml.CRY(weights[18], wires=[wires[3], wires[1]])
        qml.CRY(weights[19], wires=[wires[3], wires[2]])
