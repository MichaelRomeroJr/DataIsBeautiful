import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.input = nn.Linear(1,256)
        self.hidden_1 = nn.Linear(256,256)
        self.hidden_2 = nn.Linear(256,256)
        self.out = nn.Linear(256,1)
    
    def forward(self,x):
        x = self.input(x)
        x = F.elu(x)
        x = self.hidden_1(x)
        x = F.elu(x)
        x = self.hidden_2(x)
        x = F.elu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


def train_network(epochs):
    network = NeuralNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr = 3e-4)
    loss_fn = torch.nn.MSELoss()
    batch_size = 128
    losses = list()
        
    for epoch in range(epochs):
        if epoch%500 == 0 and epoch != 0:
            print('Epoch: '+ str(epoch))
            print('Last error: ' + str(losses[-1]))

        #save images
        inputs = np.arange(2*(-np.pi),2*(np.pi),0.1) # start,stop,step
        true_sin = np.sin(inputs)
              
        with torch.no_grad():
            net_sin = network(torch.Tensor(inputs).view(-1,1)).squeeze().numpy()
               
        fig = plt.figure(figsize=(10,10)) # outputs a lot of empty Figures 
        ax  = fig.add_subplot(111)
       
        plt.rcParams.update({'figure.max_open_warning': 0}) # ignore warning from opening so many plts     
        
        ax.plot(inputs, true_sin, label = 'Sine function')
        ax.plot(inputs, net_sin, label = 'Neural network')
        
        # Set ticks from "speical angles" in radians
        plt.xticks([2*(-np.pi),3*(-np.pi/2),-np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*(np.pi/2), 2*(np.pi)],
                  [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$3\pi/2$', r'$2\pi$']) 

        # set limit from -2pi, to 2pi
        plt.xlim(2*(-np.pi),2*(np.pi))

        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')           
        
        plt.legend(loc = 1)
        #plt.xlim(-10,10)
        plt.savefig('Plots/'+str(epoch)+'.png')
        plt.clf()
                               
        #create training numbers
        nums = np.random.uniform(-10,10,batch_size)
        x = torch.Tensor(nums)
        y = torch.Tensor(np.sin(nums))
        y = y.view(batch_size,-1)

        #train network
        optimizer.zero_grad()
        net_output = network(x.view(batch_size,-1))
        loss = loss_fn(y,net_output)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    plt.plot(range(len(losses)), losses)
    #plt.show()
    
def create_gif(folder):
    with imageio.get_writer('sinusgif15k.gif', mode = 'I', duration = 1/60) as writer:
        for filename in sorted(os.listdir(folder), key = len)[:1500]:
            image = imageio.imread(folder+filename)
            writer.append_data(image)

train_network(1000)
#train_network(100) # gif test
create_gif('Plots/')
