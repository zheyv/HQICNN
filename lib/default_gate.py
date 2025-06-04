import pennylane as qml

'''
默认电路
4层旋转门：RY + RZ + RY + RZ门
'''

def default_circuit(weights, wires):
    for i in range(len(wires)):
        qml.RY(weights[i][0], wires=wires[i])
        qml.RZ(weights[i][1], wires=wires[i])
        qml.RY(weights[i][2], wires=wires[i])
        qml.RZ(weights[i][3], wires=wires[i])