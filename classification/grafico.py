import matplotlib.pyplot as plt
import numpy as np


lr = [0.1,0.01,0.001]

sigma = [6,12,24]
################## PLOT #####################
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(3,3)
name = 'acc'
Name = 'Acc'
m = 0
for LR in lr:
    y = np.loadtxt('./data/{}_classical_optim_SGD_lr_{}.txt'.format(name,LR))
    grad = np.array(y)

    dx = np.arange( len( grad.mean(0) ) )

    z = np.loadtxt('./data/{}_hibrido_grad_optim_SGD_lr_{}.txt'.format(name,LR))
    lstm_grad = np.array(z)

    n = 0
    for SIGMA in sigma:
        z1 = np.loadtxt('./data/{}_hibrido_es_optim_SGD_lr_{}_sigma_pi_{}.txt'.format(name,LR,SIGMA))
        lstm_es = np.array(z1)
        ax[m][n].plot(dx, grad.mean(0), 'b-',label='GRAD')
        ax[m][n].fill_between(dx, grad.min(0), grad.max(0), color='b', alpha=0.3)
        ax[m][n] .plot(dx, lstm_grad.mean(0), 'r-',label='LL')
        ax[m][n] .fill_between(dx, lstm_grad.min(0), lstm_grad.max(0), color='r', alpha=0.3)
        ax[m][n] .plot(dx, lstm_es.mean(0), 'g-',label='LLES')
        ax[m][n].fill_between(dx, lstm_es.min(0), lstm_es.max(0), color='g', alpha=0.3)

        dy =np.array([grad.min(0).min(), lstm_grad.min(0).min(), lstm_es.min(0).min()]).min()
        

        ax[m][n].tick_params(axis='both', labelsize=15)


        if n ==0:
            ax[m][n].set_ylabel('{}'.format(Name),fontsize='20')
        if m ==2:
            ax[m][n].set_xlabel('Epochs',fontsize='20')
        ax[m][n].legend(fontsize='14')
        n+=1
    m+=1

#plt.subplots_adjust(left=0.05, right=0.9, bottom=0.1, top=0.99, wspace=0.3, hspace=0.3)

plt.show()




