import matplotlib.pyplot as plt
import numpy as np


'''
Noise = [0,0.01,0.03]
lr = 0.001

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1,3)

maximo = 0

for noise in Noise:

    dataGrad = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/grad_lr_{}.txt'.format(noise,lr))
    dataLL = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/ll_lr_{}.txt'.format(noise,lr))
    dataLLES = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/lles_lr_{}.txt'.format(noise,lr))

    grad = np.array(dataGrad)
    ll = np.array(dataLL)
    lles = np.array(dataLLES)
    dx = np.arange( len( grad.mean(0) ) )

    maxx = np.max(grad)
    if maxx>=maximo:
        maximo = maxx

    maxx = np.max(ll)
    if maxx>=maximo:
        maximo = maxx

    maxx = np.max(lles)
    if maxx>=maximo:
        maximo = maxx

    ax[0].plot(dx, grad.mean(0))
    ax[0].fill_between(dx, grad.min(0), grad.max(0), alpha=0.3)

    ax[1].plot(dx, ll.mean(0))
    ax[1].fill_between(dx, ll.min(0), ll.max(0), alpha=0.3)


    ax[2].plot(dx, lles.mean(0),label='$\lambda$={}'.format(noise))
    ax[2].fill_between(dx, lles.min(0), lles.max(0), alpha=0.3)


for i in range(3):
    ax[i].set_xlabel('Epochs')

ax[0].set_ylabel('Loss')
plt.legend()

for i in range(3):
    ax[i].set_ylim(-1.1,maximo)


ax[0].set_title('GRAD')
ax[1].set_title('LL')
ax[2].set_title('LLES')
#plt.subplots_adjust(left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.3,hspace=0.3)

plt.show()
'''


Noise = [0,0.01,0.03]

lr = 0.001
maximo = 0
for noise in Noise:

    dataGrad = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/grad_lr_{}.txt'.format(noise,lr))
    dataLL = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/ll_lr_{}.txt'.format(noise,lr))
    dataLLES = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/lles_lr_{}.txt'.format(noise,lr))

    grad = np.array(dataGrad)
    ll = np.array(dataLL)
    lles = np.array(dataLLES)

    maxx = np.max(grad)
    if maxx>=maximo:
        maximo = maxx

    maxx = np.max(ll)
    if maxx>=maximo:
        maximo = maxx

    maxx = np.max(lles)
    if maxx>=maximo:
        maximo = maxx


############## grad ###################
plt.rcParams.update({'font.size': 35})
plt.figure(figsize = (20,10))
for noise in Noise:

    data = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/grad_lr_{}.txt'.format(noise,lr))
    
    y = np.array(data)
    
    dx = np.arange( len( y.mean(0) ) )


    plt.plot(dx, y.mean(0), linewidth=3, label = '$\lambda = ${}'.format(noise))
    plt.fill_between(dx, y.min(0), y.max(0), alpha=0.55)

   
plt.title('GRAD')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(-1.1,maximo)
plt.legend()

plt.savefig('./figures2/grad_lr_{}.pdf'.format(lr))



############## ll ###################
plt.rcParams.update({'font.size': 35})
plt.figure(figsize = (20,10))
for noise in Noise:

    data = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/ll_lr_{}.txt'.format(noise,lr))
    
    y = np.array(data)
    
    dx = np.arange( len( y.mean(0) ) )


    plt.plot(dx, y.mean(0),linewidth=3, label = '$\lambda = ${}'.format(noise))
    plt.fill_between(dx, y.min(0), y.max(0), alpha=0.55)

   
plt.title('LL')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(-1.1,maximo)
plt.legend()
plt.savefig('./figures2/ll_lr_{}.pdf'.format(lr))


############## lles ###################
plt.rcParams.update({'font.size': 35})
plt.figure(figsize = (20,10))
for noise in Noise:

    data = np.loadtxt('./data_amplitude_damping_error_add_noise_0/data_nq_4_nl_4_noise_{}/lles_lr_{}.txt'.format(noise,lr))
    
    y = np.array(data)
    
    dx = np.arange( len( y.mean(0) ) )


    plt.plot(dx, y.mean(0), linewidth=3, label = '$\lambda = ${}'.format(noise))
    plt.fill_between(dx, y.min(0), y.max(0), alpha=0.55)

   
plt.title('LLES')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(-1.1,maximo)
plt.legend()
plt.savefig('./figures2/lles_lr_{}.pdf'.format(lr))


