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

##################################################################################################################
class QuantumClass:

    def __init__(self, nq,nl , backend = AerSimulator(), shot=1024):
        self.nq = nq
        self.nl = nl
        self.shot = shot
        self.backend = backend



        self.imput = { k : Parameter('imput{}'.format(k)) for k in range(self.nq) }


        self.theta = { k : Parameter('theta{}'.format(k)) for k in range(self.nq*self.nl) }


        self.q = QuantumRegister(self.nq)
        self.c = ClassicalRegister(self.nq)
        self.qc = QuantumCircuit(self.q,self.c)


        for i in range(self.nq):
            self.qc.ry( 2*self.imput[i] , self.q[i] )

        for i in range(self.nl):
            for j in range(self.nq):
                self.qc.ry( self.theta[i*self.nq+j] , self.q[j] )
            for j in range(self.nq-1):
                self.qc.cx(self.q[j],self.q[j+1])


        self.qc.measure(self.q,self.c)

    def E(self,nc):
        d1 = np.array([[1,-1]])
        d = np.array([[1,-1]])
        for i in range(nc-1):
            d = np.kron(d,d1)
        return d

    def run(self,imput,theta):


        imput = imput.reshape(self.nq)
        params = { self.imput[k] : imput[k].item() for k in range(self.nq) }

        theta = theta.reshape(self.nq*self.nl)
        params1 = { self.theta[k] : theta[k].item() for k in range(self.nq*self.nl) }
        params.update(params1)

        qobj = assemble(self.qc,shots=self.shot, parameter_binds = [ params ])

        job = self.backend.run(qobj)



        re = job.result().get_counts()
        prob = torch.zeros(1,2**self.nq)
        for i in re:
          prob[0][int(i,2)] = re[i]/self.shot

        H = self.E(self.nq)
        soma = 0
        for i in range(2**self.nq):
          soma+= H[0][i]*prob[0][i]
        return soma.reshape(1,1)


##################################################################################################################
class TorchCircuit_NES(Function):
    @staticmethod
    def forward(self, imput ,theta , quantumcircuit, sigma = m.pi/24  ):
        self.quantumcircuit = quantumcircuit
        result = self.quantumcircuit.run(imput,theta)
        self.n_qubit = self.quantumcircuit.nq
        self.layer = self.quantumcircuit.nl

        self.sigma = sigma

        self.save_for_backward(result,imput, theta)


        return result.float()

    @staticmethod
    def backward(self, grad_output):

        forward_tensor,imput1, theta1 = self.saved_tensors


        theta1 = theta1.reshape(self.n_qubit*self.layer)

        sigma = self.sigma

        l = int( (4+3*np.log10(self.n_qubit*self.layer))  )

        media1 = 0
        soma1 = 0

        mm1 = torch.distributions.multivariate_normal.MultivariateNormal(theta1,torch.eye( len(theta1) )*(sigma**2) )
        w1 = theta1.reshape(len(theta1),1)

        for k in range( l ):
            xi1 = mm1.sample().reshape(self.n_qubit*self.layer,1)
            d0 =  self.quantumcircuit.run(imput1,xi1)
            xi11 = 2*w1-xi1
            d1 =  self.quantumcircuit.run(imput1,xi11)
            dd = (xi1-w1)
            soma1+= (d0-d1)*dd


        media1 = soma1/(2*l*sigma**2)
        media1 = media1.float()
        result = torch.matmul( media1, grad_output.T)
        result = result.reshape(self.n_qubit*self.layer)


        return  None,result,None,None,None


##################################################################################################################
class TorchCircuit(Function):
    @staticmethod
    def forward(self, imput ,theta , quantumcircuit ):
        self.quantumcircuit = quantumcircuit
        exp_value = self.quantumcircuit.run(imput,theta)
        result = exp_value
        self.nq = self.quantumcircuit.nq
        self.nl = self.quantumcircuit.nl


        self.save_for_backward(result,imput, theta)

        return result

    @staticmethod
    def backward(self, grad_output):

        forward_tensor,imput1, theta1 = self.saved_tensors

        ######################## derivada parametros da rede ############
        input_numbers = theta1

        gradients = torch.Tensor()
        for k in range(len(theta1)):
            shift_right = input_numbers.detach().clone()
            shift_right[k] = shift_right[k] + m.pi/2
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - m.pi/2

            expectation_right = self.quantumcircuit.run(imput1,shift_right)
            expectation_left  = self.quantumcircuit.run(imput1,shift_left)

            gradient =(expectation_right - expectation_left)*(1/2)

            gradients = torch.cat((gradients, gradient.float()))


        return None,(gradients * grad_output.float()).T, None



##################################################################################################################
class model1(nn.Module):


    def __init__(self,n_qubit,n_layer,
        backend=AerSimulator(),
        shots = 1024):
        super(model1, self).__init__()

        self.quantum_circuit = QuantumClass(n_qubit,n_layer,backend,shots)
        self.alfa = torch.nn.Parameter(torch.FloatTensor(n_qubit*n_layer).uniform_(-m.pi, m.pi))



    def forward(self,input):

        return TorchCircuit.apply( input,self.alfa,self.quantum_circuit )



##################################################################################################################
class Qlayer(nn.Module):


    def __init__(self,n_qubit,n_layer,NN,backend=AerSimulator(),
        shots = 1024,sigma=m.pi/24):
        super(Qlayer, self).__init__()

        self.quantum_circuit = QuantumClass(n_qubit,n_layer,backend,shots)

        self.NN = NN
        self.sigma = sigma

    def forward(self,input,alfa):
        if self.NN == 0:
            return TorchCircuit.apply( input,alfa,self.quantum_circuit )
        elif self.NN == 1:
            return TorchCircuit_NES.apply( input,alfa,self.quantum_circuit,self.sigma )



##################################################################################################################

class model2(nn.Module):
    def __init__(self,nq,nl,NN,d_LSTM,sigma=m.pi/24):
        super(model2, self).__init__()

        self.nq = nq
        self.nl = nl
        self.out = 1
        self.qlayer = Qlayer(nq,nl,NN,sigma=sigma)
        self.dn = d_LSTM

        self.lstm = nn.LSTMCell(self.out, self.nq*self.nl)

        self.init_loss = torch.nn.Parameter(torch.FloatTensor(1, self.out).uniform_(-m.pi, m.pi))
        self.init_hx = torch.nn.Parameter(torch.FloatTensor(1, self.nq*self.nl).uniform_(-m.pi, m.pi))
        self.init_cx = torch.nn.Parameter(torch.FloatTensor(1, self.nq*self.nl).uniform_(-m.pi, m.pi))


    def forward(self, x):

        soma = 0
        hx,new_Param = self.lstm( self.init_loss, (self.init_hx,self.init_cx) )
        Param = new_Param.reshape(self.nq*self.nl)
        new_Loss = self.qlayer(x,Param)
        soma+= new_Loss.float()



        for i in range(self.dn-1):
            hx,new_Param = self.lstm( new_Loss.reshape(1,self.out).float(), (hx,new_Param) )
            Param = new_Param.reshape(self.nq*self.nl)
            new_Loss = self.qlayer(x,Param)
            soma+= new_Loss.float()


        return soma/self.dn


##################################################################################################################

def train( model, nq,nl,lr, epochs, optim_name,Nmodel , sigma):
    optimizer = optim.SGD(model.parameters(), lr=lr)

    data_hist = []
    tp = trange(epochs)
    x = torch.ones(nq)*(m.pi/4)
    for n in tp:
        tp.set_description(f" Model: {Nmodel+1} nq:{nq} nl:{nl} {optim_name} lr: {lr} sigma: pi/{sigma}   ")

        optimizer.zero_grad()
        out = model(x)
        l = out
        l.backward()
        optimizer.step()
        data_hist.append( l.item() )
    return np.array(data_hist)



##################################################################################################################
NQ =  [8]
NL =   [4,8]
epochs =1000
N_model = 5
n_LSTM = 2
LR = [0.1,0.01,0.001]

SIGMA = [6,12,24]

for nq in NQ:
    for nl in NL:

        if not os.path.exists('./data_nq_{}_nl_{}'.format(nq,nl)):
            os.mkdir('./data_nq_{}_nl_{}'.format(nq,nl))

        for lr in LR:
            ############### GRAD #################
            classica = []
            for i in range(N_model):
                net = model1(nq,nl)
                y=train( net, nq,nl,lr, epochs, 'GRAD',i , 0)
                classica.append(y)
            np.savetxt('./data_nq_{}_nl_{}/grad_lr_{}.txt'.format(nq,nl,lr),classica)
            print('')
            print('')
            print('')

            ############### LL #################

            hibrido_1 = []
            for i in range(N_model):
                net = model2(nq,nl,0,n_LSTM,0)
                z=train( net, nq,nl,lr, epochs, 'LL',i , 0)
                hibrido_1.append(z)
            np.savetxt('./data_nq_{}_nl_{}/lstm_grad_lr_{}.txt'.format(nq,nl,lr),hibrido_1)
            print('')
            print('')
            print('')
            ############### LLES #################
            for sigma in SIGMA:
                hibrido_2 = []
                for i in range(N_model):
                    net = model2(nq,nl,1,n_LSTM,m.pi/sigma)
                    z1=train( net, nq,nl,lr, epochs, 'LLES',i , sigma)
                    hibrido_2.append(z1)
                np.savetxt('./data_nq_{}_nl_{}/lstm_es_lr_{}_sigma_pi_{}.txt'.format(nq,nl,lr,sigma),hibrido_2)
                print('')
                print('')
                print('')








