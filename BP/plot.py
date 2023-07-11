import numpy as np
import matplotlib.pyplot as plt
import matplotlib

name = 'data_nl_4'
Name_xlabel = 'Qubits'

grad = np.loadtxt('./{}/var_model1.txt'.format(name))

grad = np.array(grad)

NN_ = [2,4,6,8,10,12,14]
#NN_ = [2,20,40,60,80,100]

x = np.array( NN_ )


lles_12 = np.loadtxt('./{}/var_model2_sigma_pi_12.txt'.format(name))
lles_24 = np.loadtxt('./{}/var_model2_sigma_pi_24.txt'.format(name))


matplotlib.rcParams.update({'font.size': 16})

# Cria uma figura e subplots principais
fig, axs = plt.subplots(1, 3)

# Subplot 1
ax_main_1 = axs[0]
ax_main_1.plot(x, grad,'-o',color='k',label='$\delta_{grad}$')

sigma = 6
lles = np.loadtxt('./{}/var_model2_sigma_pi_{}.txt'.format(name,sigma))
#for i in range(lles.shape[1]):
ax_main_1.plot(x, lles.T[0],'--x' ,color='b',label=' $\delta_{0}$')
ax_main_1.plot(x, lles.T[1],'--x' ,color='g',label=' $\delta_{1}$')
ax_main_1.plot(x, lles.T[2],'--x' ,color='r',label=' $\delta_{2}$')
ax_main_1.plot(x, lles.T[3],'--x' ,color='c',label=' $\delta_{3}$')
ax_main_1.plot(x, lles.T[4],'--x' ,color='y',label=' $\delta_{4}$')
ax_main_1.set_xlabel('{}'.format(Name_xlabel),fontsize=15)
ax_main_1.set_ylabel('Var',fontsize=15)


ax_main_1.set_title('$\sigma = \pi/{} $'.format(sigma))
ax_main_1.legend()
ax_main_1.set_xticks(NN_)
# Subplot menor dentro do Subplot 1
ax_inner_1 = ax_main_1.inset_axes([0.2, 0.4, 0.4, 0.3])  # Posição e tamanho do subplot menor
ax_inner_1.plot(x, grad,'-o',color='k')
ax_inner_1.plot(x,  lles.T[3],'--x' ,color='c')
ax_inner_1.plot(x,  lles.T[4],'--x',color='y' )
ax_inner_1.set_xticks(NN_)
# Subplot 2
ax_main_1 = axs[1]
ax_main_1.plot(x, grad,'-o',color='k',label='$\delta_{grad}$')

sigma = 12
lles = np.loadtxt('./{}/var_model2_sigma_pi_{}.txt'.format(name,sigma))
#for i in range(lles.shape[1]):
ax_main_1.plot(x, lles.T[0],'--x' ,color='b',label=' $\delta_{0}$')
ax_main_1.plot(x, lles.T[1],'--x' ,color='g',label=' $\delta_{1}$')
ax_main_1.plot(x, lles.T[2],'--x' ,color='r',label=' $\delta_{2}$')
ax_main_1.plot(x, lles.T[3],'--x' ,color='c',label=' $\delta_{3}$')
ax_main_1.plot(x, lles.T[4],'--x' ,color='y',label=' $\delta_{4}$')
ax_main_1.set_xlabel('{}'.format(Name_xlabel),fontsize=15)
#ax_main_1.set_ylabel('Var',fontsize=15)


ax_main_1.set_title('$\sigma = \pi/{} $'.format(sigma))
ax_main_1.legend()
ax_main_1.set_xticks(NN_)
# Subplot menor dentro do Subplot 1
ax_inner_1 = ax_main_1.inset_axes([0.2, 0.4, 0.4, 0.3])  # Posição e tamanho do subplot menor
ax_inner_1.plot(x, grad,'-o',color='k')
ax_inner_1.plot(x,  lles.T[3],'--x' ,color='c')
ax_inner_1.plot(x,  lles.T[4],'--x',color='y' )
ax_inner_1.set_xticks(NN_)
# Subplot 3
ax_main_1 = axs[2]
ax_main_1.plot(x, grad,'-o',color='k',label='$\delta_{grad}$')

sigma = 24
lles = np.loadtxt('./{}/var_model2_sigma_pi_{}.txt'.format(name,sigma))
#for i in range(lles.shape[1]):
ax_main_1.plot(x, lles.T[0],'--x' ,color='b',label=' $\delta_{0}$')
ax_main_1.plot(x, lles.T[1],'--x' ,color='g',label=' $\delta_{1}$')
ax_main_1.plot(x, lles.T[2],'--x' ,color='r',label=' $\delta_{2}$')
ax_main_1.plot(x, lles.T[3],'--x' ,color='c',label=' $\delta_{3}$')
ax_main_1.plot(x, lles.T[4],'--x' ,color='y',label=' $\delta_{4}$')
ax_main_1.set_xlabel('{}'.format(Name_xlabel),fontsize=15)
#ax_main_1.set_ylabel('Var',fontsize=15)


ax_main_1.set_title('$\sigma = \pi/{} $'.format(sigma))
ax_main_1.legend()
ax_main_1.set_xticks(NN_)
# Subplot menor dentro do Subplot 1
ax_inner_1 = ax_main_1.inset_axes([0.2, 0.4, 0.4, 0.3])  # Posição e tamanho do subplot menor
ax_inner_1.plot(x, grad,'-o',color='k')
ax_inner_1.plot(x,  lles.T[3],'--x' ,color='c')
ax_inner_1.plot(x,  lles.T[4],'--x',color='y' )
ax_inner_1.set_xticks(NN_)

plt.show()








