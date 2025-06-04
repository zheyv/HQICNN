import pennylane as qml

'''
CB结构的U3门
'''
# RZ + CB结构CRY门
class u3_cb_1:
    def __init__(self):
        self.gate_size=3
        self.weight_size = 6
    
    def circuit(self, weights, wires):
        assert(len(wires) >= 3)
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RY(weights[2], wires=wires[2])
        qml.CRZ(weights[3], wires=[wires[2], wires[0]])
        qml.CRZ(weights[4], wires=[wires[1], wires[2]])
        qml.CRZ(weights[5], wires=[wires[0], wires[1]])

# RX + CB结构CRY门
class u3_cb_2:
    def __init__(self):
        self.gate_size=3
        self.weight_size = 6
    
    def circuit(self, weights, wires):
        assert(len(wires) >= 3)
        qml.RX(weights[0], wires=wires[0])
        qml.RX(weights[1], wires=wires[1])
        qml.RX(weights[2], wires=wires[2])
        qml.CRY(weights[3], wires=[wires[2], wires[0]])
        qml.CRY(weights[4], wires=[wires[1], wires[2]])
        qml.CRY(weights[5], wires=[wires[0], wires[1]])


# RY&RZ + AA结构CRY门 + RY&RZ
class u3_aa_1:
    def __init__(self):
        self.gate_size=3
        self.weight_size = 12

    def circuit(self, weights, wires):
        assert(len(wires) >= 3)
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])
        qml.RY(weights[2], wires=wires[2])
        qml.RZ(weights[3], wires=wires[0])
        qml.RZ(weights[4], wires=wires[1])
        qml.RZ(weights[5], wires=wires[2])
        
        # AA结构
        qml.CRY(weights[6], wires=[wires[2], wires[1]])
        qml.CRY(weights[7], wires=[wires[2], wires[0]])
        qml.CRY(weights[8], wires=[wires[1], wires[0]])
        qml.CRY(weights[9], wires=[wires[1], wires[0]])
        qml.CRY(weights[10], wires=[wires[0], wires[1]])
        qml.CRY(weights[11], wires=[wires[0], wires[2]])

