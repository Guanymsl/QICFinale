#%matplotlib inline
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
from pprint import pprint
#from qiskit import Aer, execute : for IBM
from qiskit_aer import Aer
from qiskit import transpile
from qiskit_aer import QasmSimulator
from math import pi
from qiskit import *  
import tensorflow as tf
from qutip import *
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
#from qiskit import IBMQ

# Set this to the backend you are choosing for qiskit.
# For real IBMQ Evaluation, use a provider
backend = Aer.get_backend('qasm_simulator')

print("Finish import packages")

# --------------------------------------------------
# The following section we prepare the MNIST dataset
# and normalize the dataset to be in the bound 0-1
# Following this, the data is transformed using the 
# PCA algorithm down to k dimensions 
# --------------------------------------------------
test_images,test_labels = tf.keras.datasets.mnist.load_data()
# test_images: 60000 * 28 * 28 # test_labels: 60000 * .
np.set_printoptions(linewidth=200)
pprint(test_images[0][0])
print(test_images[1])
train_images = test_images[0].reshape(60000,784) # turn to 1-D array
train_labels = test_images[1]
labels = test_images[1]
train_images = train_images/255 # normalization

# --------------------------------------------------
# ---------------- PCA Section ---------------------
# --------------------------------------------------

k=2 # = q // 2, q = total quantum bits

pca = PCA(n_components=k)
pca.fit(train_images)
pca_data = pca.transform(train_images)[:10000]
pprint(pca_data)
print(pca_data.shape)

train_labels = train_labels[:10000]
t_pca_data = pca_data.copy()

# store the parameters of descaling process
pca_descaler = [[] for _ in range(k)]

# to apply scaling s.t. each entry falls in 0 to 1
for i in range(k):
    pca_descaler[i].append(pca_data[:,i].min())
    if pca_data[:,i].min() < 0:
        pca_data[:,i] += np.abs(pca_data[:,i].min())
    else:
        pca_data[:,i] -= pca_data[:,i].min()
    pca_descaler[i].append(pca_data[:,i].max())
    pca_data[:,i] /= pca_data[:,i].max()

# --------------------------------------------------
# -----  Transform PCA data to rotations ----------
# --------------------------------------------------

# the rotation angle = 2 * sin^-1 (entry^2)
# Compute rotation angles from PCA data
pca_data_rot = 2 * np.arcsin(np.sqrt(pca_data))
print(pca_data_rot.shape)

# Select data where label is 9 or 3
valid_mask = (train_labels == 7)
print(valid_mask)

# Apply the mask to PCA data
pca_data_rot = pca_data_rot[valid_mask]
print(pca_data_rot.shape)
pca_data = pca_data[valid_mask]

# Optional: also filter labels if you need them
valid_labels = train_labels[valid_mask]
print(valid_labels)

# Print explained variance
print(f"The Total Explained Variance of {k} Dimensions is {sum(pca.explained_variance_ratio_).round(3)}")

# --------------------------------------------------
# Define a function that can take in PCA'ed data and return an image
# --------------------------------------------------
def descale_points(d_point,scales=pca_descaler,tfrm=pca):
    for col in range(d_point.shape[1]):
        d_point[:,col] *= scales[col][1]
        d_point[:,col] += scales[col][0]
    reconstruction = tfrm.inverse_transform(d_point)
    return reconstruction

# Qubits Encoding 1 Dimension of Data
#All functions needed for the functionality of the circuit simulation
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      dp = np.array((predictions[i] * 127.5) + 127.5).astype('uint8')
      #plt.imshow(dp)
      plt.axis('off')
  plt.savefig('./Result/Images/image_at_epoch_{:04d}.png'.format(epoch))


def ran_ang():
    # return np.pi/2, a random angle btw 0 and pi
    return np.random.rand()*np.pi

def single_qubit_unitary(circ_ident,qubit_index,values):
    # Impose R_y(values[0]) on the qubit_index-th qubit of circ_ident
    circ_ident.ry(values[0],qubit_index)

def dual_qubit_unitary(circ_ident,qubit_1,qubit_2,values):
    # Impose R_yy(values[0]) on the qubit_1, qubit_2-th qubits (entangled) of circ_ident
    circ_ident.ryy(values[0],qubit_1,qubit_2)

def controlled_dual_qubit_unitary(circ_ident,control_qubit,act_qubit,values):
    # c-Ry
    circ_ident.cry(values[0],control_qubit,act_qubit)
    #circ_ident.cry(values[0],act_qubit,control_qubit)
    
def traditional_learning_layer(circ_ident,num_qubits,values,style="Dual",qubit_start=1,qubit_end=5):
    if style == "Dual":
        # [quibit_start, qubit_end)
        # U_y -> U_yy
        for qub in np.arange(qubit_start,qubit_end):
            single_qubit_unitary(circ_ident,qub,values[str(qub)])
        for qub in np.arange(qubit_start,qubit_end-1):
            dual_qubit_unitary(circ_ident,qub,qub+1,values[str(qub)+","+str(qub+1)])
    elif style =="Single":
        # U_y
        for qub in np.arange(qubit_start,qubit_end):
            single_qubit_unitary(circ_ident,qub,values[str(qub)])
    elif style=="Controlled-Dual":
        # U_y -> U_yy -> c-Ry
        for qub in np.arange(qubit_start,qubit_end):
            single_qubit_unitary(circ_ident,qub,values[str(qub)])
        for qub in np.arange(qubit_start,qubit_end-1):
            dual_qubit_unitary(circ_ident,qub,qub+1,values[str(qub)+","+str(qub+1)])
        for qub in np.arange(qubit_start,qubit_end-1):
            controlled_dual_qubit_unitary(circ_ident,qub,qub+1,values[str(qub)+"--"+str(qub+1)])

def data_loading_circuit(circ_ident,num_qubits,values,qubit_start=1,qubit_end=5):
    # Impose R_y(theta), while theta corresponds to the angle mapped from the data itself
    k = 0
    for qub in np.arange(qubit_start,qubit_end):
        circ_ident.ry(values[k],qub)
        k += 1

def swap_test(circ_ident,num_qubits):
    # swap_test: fidelity = P(Measurement = 0) = 1/2 + |<\phi, \psi>|^2/2
    num_swap = num_qubits//2 # exclude the anicila, half of total qubits
    for i in range(num_swap): # 0, 1, 2, ... , num_swap-1
        circ_ident.cswap(0,i+1,i+num_swap+1) # 0 qubit as the control bit
    circ_ident.h(0)
    # circuit.measure(qubit_index, classical_bit_index)
    circ_ident.measure(0,0)
        
def init_random_variables(q,style):
    trainable_variables = {} # a dictionary [str, list[float]]
    if style=="Single":
        for i in np.arange(1,q+1): # i = 1 to q
            trainable_variables[str(i)] = [ran_ang()]
    elif style=="Dual":
        for i in np.arange(1,q+1):
            trainable_variables[str(i)] = [ran_ang()]
            if i != q: # for R_yy
                trainable_variables[str(i)+","+str(i+1)] = [ran_ang()]
    elif style=="Controlled-Dual":
        for i in np.arange(1,q+1):
            trainable_variables[str(i)] = [ran_ang()]
            if i != q: # for R_yy and c-Ry
                trainable_variables[str(i)+","+str(i+1)] = [ran_ang()]
                trainable_variables[str(i)+"--"+str(i+1)] = [ran_ang()]
    return trainable_variables
    
def get_probabilities(circ,counts=5000):
    # In nutshell, this is to calculate the dot product of \phi and \psi
    circuit = transpile(circ, backend)
    job = backend.run(circuit, shots=counts)
    results = job.result().get_counts(circ)
    try:
        prob = results['0']/(results['1']+results['0'])
        prob = (prob-0.5) # since swap test prob = 0.5 -> dot product = 0, prob = 1 -> dot product = 1
        if prob <= 0.005:
            prob = 0.005
        else:
            prob = prob*2
    except:
        prob = 1
    return prob
        
# Define loss function. SWAP Test returns probability, so minmax probability is logical
def cost_function(p,yreal,trimming):
    # to evaluate p and yreal
    if yreal == 0: # this is fake data
        return -np.log(p)
        #return 1-p
    elif yreal == 1: # this is real data
        return -np.log(1-p)
        #return p
    
def generator_cost_function(p):
    # the yreal == 0 case, fake data
    return -np.log(p)

def update_weights(init_value,lr,grad):
    # lr: learning rate
    while lr*grad > 2*np.pi:
        lr /= 10
        print("Warning - Gradient taking steps that are very large. Drop learning rate")
    weight_update = lr*grad
    new_value = init_value
    print("Updating with a new value of " + str(weight_update))
    if new_value-weight_update > 2*np.pi:
        new_value = (new_value-weight_update) - 2*np.pi
    elif new_value-weight_update < 0:
        new_value = (new_value-weight_update) + 2*np.pi
    else:
        new_value = new_value - weight_update
    return new_value 

# ------------------------------------------------------------------------------------
# We treat the first n qubits are the discriminators state. n is always defined as the
# integer division floor of the qubit count.

# total qubit count = 1 + n (discriminator states) + n ()

# This is due to the fact that a state will always be k qubits, therefore the 
# number of total qubits must be 2k+1. 2k as we need k for the disc, and k to represent
# either the other learned quantum state, or k to represent a data point
# then +1 to perform the SWAP test. Therefore, we know that we will always end up
# with an odd number of qubits. We take the floor to solve for k. 1st k represents 
# disc, 2nd k represents the "loaded" state be it gen or real data
# ------------------------------------------------------------------------------------
# Use different function calls to represent training a GENERATOR or training a DISCRIMINATOR
# ------------------------------------------------------------------------------------
# THIS SECTION IS FOR THE ONLINE GENERATION OF QUANTUM CIRCUITS

q = 5
c = 1
layer_style = "Controlled-Dual"
train_var = init_random_variables(q-1,layer_style) # all are random rotation angles

# Sample = 1: for generator generate fake data / Sample = 0: for discriminator x generator do swap test
# key: Single: x/ Dual: x, y/ Controlled dual: x--y ; key_value: 0, 1, 2, 3... (training process?)
def disc_fake_training_circuit(trainable_variables,key,key_value,par_shift,diff=False,fwd_diff = False,Sample=False):
    # if now sampling, then only need half number of the quantum states
    if Sample:
        z = q//2
        circ = QuantumCircuit(q,z) # only needs to output half
    else:
        circ = QuantumCircuit(q,c)
    circ.h(0) # first quantum state for swap test
    # do gradient estimation
    if diff == True and fwd_diff == True: # prepare for f(\theta + \delta)
        trainable_variables[key][key_value] += par_shift
    if diff == True and fwd_diff == False: # prepare for f(\theta - \delta)
        trainable_variables[key][key_value] -= par_shift
    traditional_learning_layer(circ,q,trainable_variables,style=layer_style,qubit_start=1,qubit_end=q//2 +1)
    traditional_learning_layer(circ,q,trainable_variables,style=layer_style,qubit_start=q//2 +1,qubit_end=q)
    if Sample:
        for qub in range(q//2):
            circ.measure(q//2 + 1 + qub,qub) # since generator is put in the lower half of quanntum states
    else:
        swap_test(circ,q)
    if diff == True and fwd_diff == True: # recover the parameters
        trainable_variables[key][key_value] -= par_shift
    if diff == True and fwd_diff == False: # recover the parameters
        trainable_variables[key][key_value] += par_shift
    # return the whole circuit
    return circ


# discriminator x real data
# no need sampling, the remaining logic is the same
def disc_real_training_circuit(training_variables,data,key,key_value,par_shift,diff,fwd_diff):
    circ = QuantumCircuit(q,c)
    circ.h(0)
    if diff == True & fwd_diff == True:
        training_variables[key][key_value] += par_shift
    if diff == True & fwd_diff == False:
        training_variables[key][key_value] -= par_shift
    traditional_learning_layer(circ,q,training_variables,style=layer_style,qubit_start=1,qubit_end=q//2 +1)
    data_loading_circuit(circ,q,data,qubit_start=q//2 +1,qubit_end=q)
    if diff == True & fwd_diff == True:
        training_variables[key][key_value] -= par_shift
    if diff == True & fwd_diff == False:
        training_variables[key][key_value] += par_shift
    swap_test(circ,q)
    return circ

#===============================================================================

def generate_kl_divergence_hist(actual_data, epoch_results_data):
    plt.clf() # clears current figure
    sns.set()
    kl_div_vec = []
    for kl_dim in range(actual_data.shape[1]):
        kl_div = kl_divergence(actual_data[:,kl_dim],epoch_results_data[:,kl_dim])
        kl_div_vec.append(kl_div)
    return kl_div_vec

def bin_data(dataset):
    bins = np.zeros(10)
    for point in dataset:
        indx = int(str(point).split('.')[-1][0]) # The shittest way imaginable to extract the first val aft decimal
        bins[indx] +=1  
    bins /= sum(bins)
    return bins

def kl_divergence(p_dist, q_dist):
    p = bin_data(p_dist)
    q = bin_data(q_dist)
    kldiv = 0
    for p_point,q_point in zip(p,q):
        kldiv += (np.sqrt(p_point) - np.sqrt(q_point))**2
    kldiv = (1/np.sqrt(2))*kldiv**0.5 
    return kldiv
    #return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))  # ?... are we confident in this... 

#================================================================================
    
# Checkpointing code
def save_variables(var_dict,epoch):
    with open(f"Epoch-{epoch}-Variables-numbers-10",'w') as file:
        file.write(str(train_var))

tracked_kl_div_1 = []
tracked_kl_div_2 = []

circ = QuantumCircuit(q,c)
circ.h(0)

#q = 5
#c = 1
#layer_style = "Controlled-Dual"
#train_var = init_random_variables(q-1,layer_style) # all are random rotation angles

# Initial Learning Settings such as alpha etc.
tracked_d_loss = []
gradients = []
learning_rate=0.01
train_iter = 250
tracked_g_loss = []
gradients_g = []
corr = 0
wrong= 0 
loss_d_to_g = 0
loss_d_to_real = 0
tracked_loss_d_to_g = []
tracked_loss_d_to_real = []
train_on_fake = 5
df = [0,0]
print('Starting Training')
print('-'*20)

for epoch in np.arange(1,100):
    par_shift = 0.5*np.pi*np.sqrt(1/(epoch+1))
    # ------------------------------------------------------------------------------------------
    # This section is the discriminator training section
    # Each data point is tested against a random number, of which it decidesa wheter to 
    # Train against discerning between fake or real 
    # This causes "unstable" loss functions, but not very "unstable". Just slightly inconsistent
    # ------------------------------------------------------------------------------------------
    counter = 0
    for _ in range(1):
        for key,value in train_var.items():
            if str(q//2 + 1 ) in key:
                break # ignore the down half parameter
            for key_value in range(len(value)):
                # def disc_fake_training_circuit(trainable_variables,key,key_value,par_shift,diff=False,fwd_diff = False,Sample=False)
                # Why yreal = 1?
                forward_diff = cost_function(get_probabilities(disc_fake_training_circuit(train_var,key,key_value,par_shift,diff=True,fwd_diff=True)),1,None)
                backward_diff = cost_function(get_probabilities(disc_fake_training_circuit(train_var,key,key_value,par_shift,diff=True,fwd_diff=False)),1,None)
                df = 0.5*(forward_diff-backward_diff)
                if abs(df)>1:
                    df = df/abs(df)
                # update train varaiable
                train_var[key][key_value] -= df*learning_rate/10
    print("Finish discriminator training on fake data")
    for index,point in tqdm(enumerate(pca_data_rot), total=len(pca_data_rot), desc="Training discriminator"):
        #print(index, point)
        df = [0,0]
        gradients = []
        loss= [0,0]
        #Training the Discriminator:
        for key,value in train_var.items():
            if str(q//2 + 1) in key:
                break
            for key_value in range(len(value)):
                #TRAIN ON REAL DATA
                # BETIS HERE
                # _________
                # why yreal = 0
                forward_diff = cost_function(get_probabilities(disc_real_training_circuit(train_var,point,key,key_value,par_shift,diff=True,fwd_diff=True)),0,None)
                backward_diff = cost_function(get_probabilities(disc_real_training_circuit(train_var,point,key,key_value,par_shift,diff=True,fwd_diff=False)),0,None)
                df = 0.5*(forward_diff-backward_diff)
                train_var[key][key_value] -= learning_rate*df
        loss[0] += cost_function(get_probabilities(disc_real_training_circuit(train_var,point,key,key_value,par_shift,diff=False,fwd_diff=False)),0,None)
        loss[1] += 1
    print("Finish discriminator training on real data")
    loss_g = [0,0]
    # ------------------------------------------------------------------------------------------
    # This section is the generator training section
    # The discriminator just looks to fool the state we learnt above 
    # This means that instead of learning 10000 times, we could up the learning rate and just learn a few more times
    # We dont want it to be too large so it spins around the qubits state
    # ------------------------------------------------------------------------------------------
    #Train the generator now as much as we trained the Disc
    for _ in tqdm(range(len(pca_data_rot)//10), desc="Generator Training"):
        gen_params=True
        for key,value in train_var.items():
            if str(q//2 + 1) not in key and gen_params:
                #print(f"{key} is not a GAN parameter")
                continue
            else: 
                gen_params = False
            for key_value in range(len(value)):
                #TRAIN ON FAKE DATA
                forward_diff = generator_cost_function(get_probabilities(disc_fake_training_circuit(train_var,key,key_value,par_shift,diff=True,fwd_diff=True)))
                backward_diff = generator_cost_function(get_probabilities(disc_fake_training_circuit(train_var,key,key_value,par_shift,diff=True,fwd_diff=False)))
                # 1/2 * (f(\theta + \delta) - f(\theta - \delta))
                df = 0.5*(forward_diff-backward_diff)
                train_var[key][key_value] -= df*learning_rate*2.5
        loss_g[0] += generator_cost_function(get_probabilities(disc_fake_training_circuit(train_var,key,key_value,par_shift,diff=False,fwd_diff=False)))
        loss_g[1] +=1
    print(f"Generator Loss: {loss_g[0]/loss_g[1]}")
    # tracked_g_loss record the "AVERAGE" generator loss
    tracked_g_loss.append(loss_g[0]/loss_g[1])

    loss_qgan = cost_function(get_probabilities(disc_fake_training_circuit(train_var,key,key_value,par_shift,diff=False,fwd_diff=False)),1,None) 
    t_loss = loss_qgan + (loss[0]/loss[1])
    tracked_loss_d_to_real.append(loss[0]/loss[1])
    tracked_loss_d_to_g.append(loss_qgan)
    print(f"Discriminator Loss: {t_loss}")

    tracked_d_loss.append(t_loss)
    print("-"*20)

    #=============================================================================================

    data = []
    circ = disc_fake_training_circuit(train_var,point,key,key_value,par_shift,Sample=True)
    new_circ = transpile(circ, backend)
    n_results = q//2
    for _ in range(500):
        job = backend.run(new_circ, shots=20)
        #job = execute(circ, backend, shots=20)
        results = job.result().get_counts(circ)
        bins = [[0,0] for _ in range(n_results)]
        for key,value in results.items():
            for i in range(n_results):
                if key[-i-1]== '1':
                    bins[i][0] += value
                bins[i][1] += value
        for i,pair in enumerate(bins):
            bins[i]= pair[0]/pair[1]
        data.append(bins)
    data = np.array(data)
    try:
        graph = sns.jointplot(x=data[:,0],y=data[:,1],kind="kde",ylim=(0,1),xlim=(0,1))
        graph.x = pca_data[:,0]
        graph.y = pca_data[:,1]
        graph.plot_joint(plt.scatter, marker='o', c='r', s=5)
        #plt.show()
    except:
        pass
    plt.savefig("./Report/qgan_ICLR_-epoch-mnist-{}-generated-distribution".format(epoch))
    dim1_kl_div = generate_kl_divergence_hist(pca_data, data)
    print(dim1_kl_div)
    tracked_kl_div_1.append(np.mean(np.array(dim1_kl_div)))
    print(tracked_kl_div_1)

    # For accurate KL Div we need to usue higher shots
    data = []
    for _ in range(16):
        job = backend.run(new_circ, shots=20)
        #job = execute(circ, backend, shots=20)
        results = job.result().get_counts(circ)
        bins = [[0,0] for _ in range(n_results)]
        for key,value in results.items():
            for i in range(n_results):
                if key[-i-1]== '1':
                    bins[i][0] += value
                bins[i][1] += value
        for i,pair in enumerate(bins):
            bins[i]= pair[0]/pair[1]
        data.append(bins)
    data = np.array(data)
    new_info = descale_points(data[:16])
    new_info = new_info.reshape(new_info.shape[0],28,28)
    print(f"Epoch {epoch} Generated Images")
    for i in range(new_info.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(new_info[i, :, :], cmap='gray')
        plt.axis('off')
    plt.savefig("./Report/qgan_ICLR_-epoch-mnist-{}-generated-images".format(epoch))
    #plt.show()
    with open('new_qgan_results_mnis_epoch_ICLR_{}.txt'.format(epoch), 'w') as file:
        file.write("Tracked KL Divergence\n")
        file.write(str(tracked_kl_div_1)+"\n")
        file.write("Loss Of Generator\n")
        file.write(str(tracked_g_loss)+"\n")
        file.write("Loss Of Discriminator\n")
        file.write(str(tracked_d_loss)+"\n")
    save_variables(train_var,epoch)