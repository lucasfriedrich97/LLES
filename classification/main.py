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

import sklearn.datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


#####################################################################################

class QuantumClass:

    def __init__(self, nq,nl, backend = AerSimulator(), shot=1024):
        self.nq = nq
        self.nl = nl
        self.shot = shot
        self.backend = backend

        self.imput = { k : Parameter('imput{}'.format(k)) for k in range(2) }
        self.theta = { k : Parameter('theta{}'.format(k)) for k in range(self.nq*self.nl) }

        self.q = QuantumRegister(self.nq)
        self.c = ClassicalRegister(1)
        self.qc = QuantumCircuit(self.q,self.c)

        for i in range(self.nq):
            self.qc.ry( self.imput[0] , self.q[i] )
            self.qc.rz( self.imput[1] , self.q[i] )

        for i in range(self.nl):
            for j in range(self.nq):
                self.qc.ry( self.theta[i*self.nq+j] , self.q[j] )
            for j in range(self.nq-1):
                self.qc.cx(self.q[j],self.q[j+1])

        self.qc.measure(self.q[-1],self.c)

    def run(self,imput,theta):

        imput = imput.reshape(2)
        params = { self.imput[k] : imput[k].item() for k in range(2) }

        theta = theta.reshape(self.nq*self.nl)
        params1 = { self.theta[k] : theta[k].item() for k in range(self.nq*self.nl) }
        params.update(params1)

        qobj = assemble(self.qc,shots=self.shot, parameter_binds = [ params ])

        job = self.backend.run(qobj)

        re = job.result().get_counts()
        try:
            return torch.tensor([re['0']/self.shot])
        except:
            return torch.tensor([0])

#####################################################################################

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

        return  None,result,None,None,None,None


#####################################################################################

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


        input_numbers = theta1

        gradients = torch.Tensor()
        for k in range(len(theta1)):
            shift_right = input_numbers.detach().clone()
            shift_right[k] = shift_right[k] + m.pi/8
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - m.pi/8

            expectation_right = self.quantumcircuit.run(imput1,shift_right)
            expectation_left  = self.quantumcircuit.run(imput1,shift_left)

            gradient =expectation_right - expectation_left

            gradients = torch.cat((gradients, gradient.float()))

        return None,(gradients * grad_output.float()).T, None


#####################################################################################

class model1(nn.Module):


    def __init__(self,n_qubit,n_layer,
        backend=AerSimulator(),
        shots = 1024):
        super(model1, self).__init__()

        self.quantum_circuit = QuantumClass(n_qubit,n_layer,backend,shots)
        self.alfa = torch.nn.Parameter(torch.FloatTensor(n_qubit*n_layer).uniform_(-m.pi, m.pi))


    def forward(self,input):

        return TorchCircuit.apply( input,self.alfa,self.quantum_circuit )


#####################################################################################

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


#####################################################################################


class model2(nn.Module):
    def __init__(self,nq,nl,NN,d_LSTM,sigma):
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
        self.Loss = nn.MSELoss()


    def forward(self, x,y=None):
        if y == None:

          soma = 0
          hx,new_Param = self.lstm( self.init_loss, (self.init_hx,self.init_cx) )
          Param = new_Param.reshape(self.nq*self.nl)
          return self.qlayer(x,Param)

        else:

          soma = 0
          hx,new_Param = self.lstm( self.init_loss, (self.init_hx,self.init_cx) )
          Param = new_Param.reshape(self.nq*self.nl)
          new_Loss = self.Loss(self.qlayer(x,Param),y)
          soma+= new_Loss.float()



          for i in range(self.dn-1):
              hx,new_Param = self.lstm( new_Loss.reshape(1,self.out).float(), (hx,new_Param) )
              Param = new_Param.reshape(self.nq*self.nl)
              new_Loss = self.Loss(self.qlayer(x,Param),y)
              soma+= new_Loss.float()


          return soma/self.dn

#####################################################################################

def data_set(nTrain,nTest,Noise,dd):

    if dd == 1:
        x_train, y_train = sklearn.datasets.make_circles(nTrain,noise=Noise, factor=0.2, random_state=1)
        x_test, y_test =  sklearn.datasets.make_circles(nTest,noise=Noise, factor=0.2, random_state=1)
    else:
        x_train, y_train = sklearn.datasets.make_moons(nTrain,noise=Noise)
        x_test, y_test =  sklearn.datasets.make_moons(nTest,noise=Noise)


    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)

    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    return (x_train,y_train),(x_test,y_test)

#####################################################################################

def validacao(model,xtest,ytest):
  soma = 0
  for i in range(len(xtest)):
    out = model(xtest[i])
    if out.item() >=0.5 and ytest[i].item() == 1:
      soma+=1
    if out.item() <0.5 and ytest[i].item() == 0:
      soma+=1
  return soma/len(xtest)

#####################################################################################

def trainModel1(xtrain,ytrain, xtest,ytest,lr,nq,nl, epochs,Nmodel,optim='SGD' ):
    loss = nn.MSELoss()
    net = model1(nq,nl)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif optim =='Adam':
         optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_ = []
    validacao_ = []

    for epoch in range(epochs):
        tp = trange(xtrain.shape[0])
        soma_loss = 0

        for i in tp:
            tp.set_description(f"Model: {Nmodel+1} lr: {lr} optim:{optim} epoch: {epoch+1}/{epochs}")
            optimizer.zero_grad()
            out = net(xtrain[i]).float()
            l = loss(out,ytrain[i].float())
            l.backward()
            optimizer.step()
            soma_loss+=l.item()

        loss_.append(soma_loss/xtrain.shape[0])
        validacao_.append( validacao(net,xtest,ytest) )

    return np.array(loss_),np.array(validacao_)


#####################################################################################

def trainModelN(xtrain,ytrain,xtest,ytest, lr,nq,nl,N,n_LSTM, epochs,Nmodel,sigma=1,optim='SGD' ):
    met = 'LSTM GRAD'
    if N ==1:
        met = 'LSTM ES'

    sigma_ = m.pi/sigma
    net = model2(nq,nl,N,n_LSTM,sigma_)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif optim =='Adam':
         optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_ = []
    validacao_ = []

    for epoch in range(epochs):
        tp = trange(xtrain.shape[0])
        soma_loss = 0

        for i in tp:
            tp.set_description(f"Model: {Nmodel+1} lr: {lr} method: {met} optim:{optim} sigma: pi/{sigma} epoch: {epoch+1}/{epochs}")
            optimizer.zero_grad()
            out = net(xtrain[i],ytrain[i].float())
            l = out
            l.backward()
            optimizer.step()
            soma_loss+=l.item()



        loss_.append(soma_loss/xtrain.shape[0])
        validacao_.append( validacao(net,xtest,ytest) )


    return np.array(loss_),np.array(validacao_)



#####################################################################################

def grafico(dx,grad,lstm,nes,nq,nl,lr,n_LSTM,sigma,name):
    grad = np.array(grad)
    lstm = np.array(lstm)
    nes = np.array(nes)
    plt.plot(dx, grad.mean(0), 'b-',label='GRAD')
    plt.fill_between(dx, grad.min(0), grad.max(0), color='b', alpha=0.3)

    plt.plot(dx, lstm.mean(0), 'r-',label='LSTM GRAD')
    plt.fill_between(dx, lstm.min(0), lstm.max(0), color='r', alpha=0.3)


    plt.plot(dx, nes.mean(0), 'g-',label='LSTM ES')
    plt.fill_between(dx, nes.min(0), nes.max(0), color='g', alpha=0.3)

    if name == 'Acc':
        plt.legend(fontsize=15,loc='lower right')
    elif name == 'Loss':
        plt.legend(fontsize=15,loc='upper right')

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('{}'.format(name),fontsize=15)
    plt.savefig('./figures/{}_lr_{}_sigma_pi_{}.pdf'.format(name,lr,sigma))
    plt.close()
#####################################################################################


if not os.path.exists('./dataTrain'):
    os.mkdir('./dataTrain')

if not os.path.exists('./data'):
    os.mkdir('./data')

if not os.path.exists('./figures'):
    os.mkdir('./figures')


(xtrain,ytrain),(xtest,ytest) = data_set(100,50,0.1,1)

np.savetxt('./dataTrain/xtrain.txt',xtrain)
np.savetxt('./dataTrain/ytrain.txt',ytrain)
np.savetxt('./dataTrain/xtest.txt',xtest)
np.savetxt('./dataTrain/ytest.txt',ytest)
#####################################################################################

nq = 4
nl = 8
epochs = 40

N_model = 4
N_LSTM = 2
OPTIM = 'SGD'


lr = [0.1,0.01,0.001]

sigma = [6,12,24]




for LR in lr:
    ############ GRAD ###################
    loss_hist_C = []
    validacao_hist_C = []
    print('')
    print('')
    print('')

    for i in range(N_model):
        y,z=trainModel1(xtrain,ytrain,xtest,ytest,LR,nq,nl,epochs,i,optim=OPTIM)
        loss_hist_C.append(y)
        validacao_hist_C.append(z)
        print('')
        print('')
        print('')
    np.savetxt('./data/loss_classical_optim_{}_lr_{}.txt'.format(OPTIM,LR),loss_hist_C)
    np.savetxt('./data/acc_classical_optim_{}_lr_{}.txt'.format(OPTIM,LR),validacao_hist_C)

    ############ LL ###################


    loss_hist_HC_1 = []
    validacao_hist_HC_1 = []

    for i in range(N_model):
        y,z=trainModelN(xtrain,ytrain,xtest,ytest, LR,nq,nl,0,N_LSTM, epochs,i,sigma=1,optim=OPTIM )
        loss_hist_HC_1.append(y)
        validacao_hist_HC_1.append(z)
        print('')
        print('')
        print('')
    np.savetxt('./data/loss_hibrido_grad_optim_{}_lr_{}.txt'.format(OPTIM,LR),loss_hist_HC_1)
    np.savetxt('./data/acc_hibrido_grad_optim_{}_lr_{}.txt'.format(OPTIM,LR),validacao_hist_HC_1)

    ############ LLES ###################

    for SIGMA in sigma:
        loss_hist_HC_2 = []
        validacao_hist_HC_2 = []


        for i in range(N_model):
            y,z=trainModelN(xtrain,ytrain,xtest,ytest, LR,nq,nl,1,N_LSTM, epochs,i,sigma=SIGMA,optim=OPTIM )
            loss_hist_HC_2.append(y)
            validacao_hist_HC_2.append(z)
            print('')
            print('')
            print('')
        np.savetxt('./data/loss_hibrido_es_optim_{}_lr_{}_sigma_pi_{}.txt'.format(OPTIM,LR,SIGMA),loss_hist_HC_2)
        np.savetxt('./data/acc_hibrido_es_optim_{}_lr_{}_sigma_pi_{}.txt'.format(OPTIM,LR,SIGMA),validacao_hist_HC_2)


        #dx = np.arange(epochs)
        #grafico(dx,loss_hist_C,loss_hist_HC_1,loss_hist_HC_2,nq,nl,LR,N_LSTM,SIGMA,'Loss')
        #grafico(dx,validacao_hist_C,validacao_hist_HC_1,validacao_hist_HC_2,nq,nl,LR,N_LSTM,SIGMA,'Acc')





