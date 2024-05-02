import numpy as np
import math as m
import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from qiskit import *
import qiskit
from qiskit import assemble,Aer
from qiskit.visualization import *
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
from collections.abc import Iterable
import functools
import os
from qiskit.providers.aer.noise import amplitude_damping_error

from qiskit.providers.aer.noise import NoiseModel
from qiskit import IBMQ, transpile


def plot_grafico_de_barras(dicionario):
	dicionario = {key: dicionario[key] for key in ['00', '11'] if key in dicionario}
	# Separar chaves e valores do dicionário
	chaves = list(dicionario.keys())
	valores = list(dicionario.values())

	# Criar o gráfico de barras
	plt.rcParams.update({'font.size': 40})
	plt.bar(chaves, valores, color='cornflowerblue', width=0.6, zorder=2)
	plt.ylabel('Probability')

	# Adicionar grade
	plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
	# Mostrar o gráfico
	plt.show()

q = QuantumRegister(2,name='q')
c = ClassicalRegister(2,name='c')
qc = QuantumCircuit(q,c)

qc.h(0)
qc.cx(0,1)

qc.measure([0,1],[0,1])


noise_model = NoiseModel()
# Perform a noise simulation
gamma = 0.2  # Taxa de amortecimento

# Criando o erro de amortecimento de amplitude para um qubit
error = amplitude_damping_error(gamma)

# Adicionando o erro ao modelo de ruído para todos os gates u1, u2, u3
noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

backend = AerSimulator(noise_model=noise_model)






transpiled_circ = transpile(qc, backend)



qobj = assemble(transpiled_circ,shots=10000)

job = backend.run(qobj)




re=job.result().get_counts()

for chave, valor in re.items():
    re[chave] = valor / 10000


plot_grafico_de_barras(re)



