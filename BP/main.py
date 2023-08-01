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


class QuantumClass:

    def __init__(self, nq,nl, backend = AerSimulator(), shot=1024):
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



    def run(self,imput,theta):


        imput = imput.reshape(self.nq)
        params = { self.imput[k] : imput[k].item() for k in range(self.nq) }

        theta = theta.reshape(self.nq*self.nl)
        params1 = { self.theta[k] : theta[k].item() for k in range(self.nq*self.nl) }
        params.update(params1)

        qobj = assemble(self.qc,shots=self.shot, parameter_binds = [ params ])

        job = self.backend.run(self.qc,qobj)



        re = job.result().get_counts()
        prob = torch.zeros(1,2**self.nq)
        for i in re:
            prob[0][int(i,2)] = re[i]/self.shot

        return prob[0][0].reshape(1,1)


class TorchCircuit_NES(Function):
    @staticmethod
    def forward(self, imput ,theta , quantumcircuit, sigma = m.pi/24 , dl=None ):
        self.quantumcircuit = quantumcircuit
        result = self.quantumcircuit.run(imput,theta)
        self.n_qubit = self.quantumcircuit.nq
        self.layer = self.quantumcircuit.nl

        self.sigma = sigma
        self.dl = dl
        self.save_for_backward(result,imput, theta)


        return result.float()

    @staticmethod
    def backward(self, grad_output):

        forward_tensor,imput1, theta1 = self.saved_tensors


        theta1 = theta1.reshape(self.n_qubit*self.layer)
        #print(theta1.shape)
        sigma = self.sigma
        if not self.dl:
            l = int( (4+3*np.log10(self.n_qubit*self.layer))  )
        else:
            l = self.dl

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


        return  None,result,None,None,None,None



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


class model1(nn.Module):


    def __init__(self,n_qubit,n_layer,
        backend=AerSimulator(),
        shots = 1024):
        super(model1, self).__init__()


        self.quantum_circuit = QuantumClass(n_qubit,n_layer,backend,shots)
        self.alfa = torch.nn.Parameter(torch.FloatTensor(n_qubit*n_layer).uniform_(-m.pi, m.pi))







    def forward(self,input):

        return TorchCircuit.apply( input,self.alfa,self.quantum_circuit )


class Qlayer(nn.Module):


    def __init__(self,n_qubit,n_layer,NN,
        backend=AerSimulator(),
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


class model2(nn.Module):
    def __init__(self,nq,nl,NN,d_LSTM,sigma):
        super(model2, self).__init__()

        self.nq = nq
        self.nl = nl
        self.out = 1
        self.qlayer = Qlayer(nq,nl,NN,sigma=sigma)
        self.dn = d_LSTM
        self.Param_ = []

        self.lstm = nn.LSTMCell(self.out, self.nq*self.nl)

        self.init_loss = torch.nn.Parameter(torch.FloatTensor(1, self.out).uniform_(-m.pi, m.pi))

        self.init_hx = torch.nn.Parameter(torch.FloatTensor(1, self.nq*self.nl).uniform_(-m.pi, m.pi))

        self.init_cx = torch.nn.Parameter(torch.FloatTensor(1, self.nq*self.nl).uniform_(-m.pi, m.pi))



    def forward(self, x):
        self.Param_ = []
        soma = 0
        hx,new_Param = self.lstm( self.init_loss, (self.init_hx,self.init_cx) )
        self.Param_.append(new_Param)
        Param = new_Param.reshape(self.nq*self.nl)
        new_Loss = self.qlayer(x,Param)
        soma+= new_Loss.float()



        for i in range(self.dn-1):
            hx,new_Param = self.lstm( new_Loss.reshape(1,self.out).float(), (hx,new_Param) )
            Param = new_Param.reshape(self.nq*self.nl)
            self.Param_.append(new_Param)
            new_Loss = self.qlayer(x,Param)
            soma+= new_Loss.float()


        return soma/self.dn


def var_model1(nq,nl,epochs):

    delta = []

    x = torch.ones(nq)*(m.pi/2)
    tp = trange(epochs)
    for i in tp:
        tp.set_description(f" Var_model1 nq:{nq} nl:{nl}  ")
        model = model1(nq,nl)# gera o modelo com variaveis aleatorias

        theta_t = model.state_dict()['alfa'] # obter o valor das variaveis em t
        l=model(x)# calcula funcao custo
        dw=torch.autograd.grad(l,model.parameters())# obtem o gradiente da funcao custo

        theta_t_1 = theta_t-dw[0] # obter o valor das variaveis em t+1


        delta.append(theta_t[0].item()-theta_t_1[0].item())#calcula a diferenca  theta_t-theta_t_1 de um parametro

    return np.var(delta)

def var_model2(nq,nl,epochs,N_LSTM, sigma):
    x = torch.ones(nq)*(m.pi/2)

    delta = []
    tp = trange(epochs)
    for i in tp:
        tp.set_description(f" Var_model2 nq:{nq} nl:{nl} N_LSTM:{N_LSTM}, sigma: pi/{sigma} ")
        model_ = model2(nq,nl,1,N_LSTM,m.pi/sigma)

        model_(x)


        dd = []
        for j in range(len(model_.Param_)-1):

            dd.append( model_.Param_[j+1][0][0].item() - model_.Param_[j][0][0].item())


        delta.append(dd)
    delta_ = np.array(delta)
    var_delta = []
    for i in range(delta_.shape[1]):

        var_delta.append( np.var(delta_.T[i]) )
    return var_delta

'''
nl = 4
N_LSTM = 6
epochs = 2000
SIGMA = [6,12,24]

model1_var = []
for nq in [2,4,6,8,10,12,14]:
    model1_var.append(var_model1(nq,nl,epochs))


np.savetxt('./data/var_model1.txt',model1_var)

for sigma in SIGMA:
    model2_var = []
    for nq in  [2,4,6,8,10,12,14]:
        model2_var.append(var_model2(nq,nl,epochs,N_LSTM,sigma))
    np.savetxt('./data/var_model2_sigma_pi_{}.txt'.format(sigma),model2_var)


'''


nq = 4
N_LSTM = 6
epochs = 2000
SIGMA = [6,12,24]

model1_var = []
for nl in [2,20,40,60,80,100]:
    model1_var.append(var_model1(nq,nl,epochs))


np.savetxt('./data/var_model1.txt',model1_var)

for sigma in SIGMA:
    model2_var = []
    for nl in  [2,20,40,60,80,100]:
        model2_var.append(var_model2(nq,nl,epochs,N_LSTM,sigma))
    np.savetxt('./data/var_model2_sigma_pi_{}.txt'.format(sigma),model2_var)

