import pennylane as qml

'''
U2门
'''

# 旋转门与CNOT门交替
class u2_1:
    def __init__(self):

        self.gate_size=2
        self.weight_size = 6

    def circuit(self, weights, wires):
        assert(len(wires) >= 2)
        qml.RZ(weights[0], wires=wires[0])
        qml.RZ(weights[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RY(weights[2], wires=wires[0])
        qml.RY(weights[3], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RZ(weights[4], wires=wires[0])
        qml.RZ(weights[5], wires=wires[1])

# 旋转门与CRy门交替
class u2_2:
    def __init__(self):
        self.gate_size=2
        self.weight_size = 9 

    def circuit(self, weights, wires):
        assert(len(wires) >= 2)
        qml.RZ(weights[0], wires[0])
        qml.RZ(weights[1], wires[1])
        qml.RY(weights[2], wires[0])
        qml.RY(weights[3], wires[1])
        qml.CRZ(weights[4], wires=[wires[0], wires[1]])
        qml.RY(weights[5], wires[0])
        qml.RY(weights[6], wires[1])
        qml.RZ(weights[7], wires[0])
        qml.RZ(weights[8], wires[1])
