
import pickle
from itertools import count
import os, sys, random,cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util.solver import Solver
from util.parser import get_parser
from util.calculate import calculate
from sklearn.decomposition import PCA
from PIL import Image
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
import json

class Env():
    def __init__(self,input='static/example_1.png',target_FSI=2.6,target_GSI=0.3,target_L=8.74,target_OSR=0.36):
        self.color_list = [(255,255,255),(68,58,130),(49,104,142),(33,144,141),(53,183,121),(143,215,68),(253,231,37)]
        self.args = get_parser().parse_args(args=['--name', 'Shenzhen',  '--mode', 'test' ,'--data_dir', 'datasets/Shenzhen',  '--color_file', 'data/Shenzhen.txt', '--img_size', '512', '--batch_size', '1'])
        self.solver = Solver(self.args)
        self.solver.load_model(latest=True)
        self.dataset = self.solver.dataset['test']
        with open(os.path.join(self.solver.args.resume_dir,'pca.pickle'),'rb') as f:
            self.pca = pickle.load(f)
        self.target_FSI = target_FSI
        self.target_GSI = target_GSI
        self.fsi_direction = np.load(r'styles/%s/FSI_direction.npy'%self.args.name)
        self.gsi_direction = np.load(r'styles/%s/GSI_direction.npy'%self.args.name)
        self.l_direction = np.load(r'styles/%s/L_direction.npy'%self.args.name)
        self.osr_direction = np.load(r'styles/%s/OSR_direction.npy'%self.args.name)
        self.seg = self.path2seg(input)
    
    def path2seg(self,input):
        seg = Image.open(input).convert('RGB')
        seg = self.dataset.encode_segmap(seg)
        seg = self.dataset.transform(Image.fromarray(seg)).unsqueeze(0)
        seg = seg.to(self.solver.device)
        return seg
        
    def generate_image(self,action):
        with torch.no_grad():
            latent_vector = self.pca.inverse_transform(action)
            latent_vector = torch.Tensor([latent_vector]).to(self.solver.device)
            gen_img = self.solver.nets.generator(self.seg,latent_vector)
            gen_img = transforms.ToPILImage()(gen_img[0])
            # gen_img.save('results/gen_img.png')
            gen_img = np.array(gen_img)
        return gen_img,latent_vector.cpu().numpy()
    
    def adjust_image(self,latent_vector,direction,coeff):
        with torch.no_grad():
            new_latent_vector = torch.Tensor(latent_vector + coeff*direction)
            new_latent_vector = new_latent_vector.to(self.solver.device)
            gen_img = self.solver.nets.generator(self.seg,new_latent_vector)
            gen_img = transforms.ToPILImage()(gen_img[0])
            gen_img = np.array(gen_img)
        return gen_img,new_latent_vector.cpu().numpy()
    
    def createmodel(self,img):
        (FSI,GSI,L,OSR),contours,heights,ids = calculate(img)
        color_list = [(255,255,255),(68,58,130),(49,104,142),(33,144,141),(53,183,121),(143,215,68),(253,231,37)]
        data = []
        d0 = dict()
        d0['coordinates'] = [[-200,-200],[-200,200],[200,200],[200,-200],[-200,-200]]
        d0['color'] = 'rgb(0,0,0)'
        d0['height']= -20
        data.append(d0)
        for c,h,i in zip(contours,heights,ids):
            d = dict()
            r,g,b = color_list[i]
            d['coordinates'] = [[x-200,200-y] for x,y  in np.squeeze(c).tolist()]
            d['color'] = 'rgb(%s,%s,%s)'%(r,g,b)
            d['height']= h
            data.append(d)
        data = json.dumps(data)
        return data,FSI,GSI,L,OSR
    
    def reset(self):
        action = np.zeros(4)
        self.img,_ = self.generate_image(action)
        data,contours,heights,ids = calculate(self.img)
        FSI,GSI,L,OSR = data
        self.state = np.array([FSI,GSI,self.target_FSI,self.target_GSI])
        return self.state

    def step(self,action):
        self.img,_ = self.generate_image(action)
        data,contours,heights,ids = calculate(self.img)
        FSI,GSI,L,OSR = data
        self.state = np.array([FSI,GSI,self.target_FSI,self.target_GSI])
        reward = -(abs(FSI - self.target_FSI)/self.target_FSI+abs(GSI-self.target_GSI)/self.target_GSI)
        done = False
        info = None
        return self.state,reward,done,info

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class DDPG(object):
    def __init__(self):
        args = Munch()
        args.tau = 0.005
        args.learning_rate = 1e-6
        args.gamma = 0.99
        args.capacity = 100
        args.batch_size = 1
        args.max_episode = 10000
        args.checkpoints_dir = './checkpoints/'
        args.log_dir = './logs'
        args.state_dim = 4
        args.action_dim= 4
        args.max_action=10
        args.exploration_noise = 1
        args.mode = 'train'
        args.load = False
        args.update_iteration = 10
        args.test_iteration = 10
        args.save_interval = 100
        args.log_interval =50
        args.max_length_of_trajectory = 200
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actor = Actor(args.state_dim, args.action_dim, args.max_action).to(self.device)
        self.actor_target = Actor(args.state_dim, args.action_dim, args.max_action).to(self.device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(args.state_dim, args.action_dim).to(self.device)
        self.critic_target = Critic(args.state_dim, args.action_dim).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer(args.capacity)
        # self.writer = SummaryWriter(args.log_dir)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        for it in range(self.args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1-d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), self.args.checkpoints_dir + 'actor.pth')
        torch.save(self.critic.state_dict(), self.args.checkpoints_dir + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(self.args.checkpoints_dir + 'actor.pth'))
        self.critic.load_state_dict(torch.load(self.args.checkpoints_dir + 'critic.pth'))
        # print("====================================")
        # print("model has been loaded...")
        # print("====================================")



def main():
    agent = DDPG()
    env =Env(input='static/example_1.png',target_FAR=3.5,target_BCR=0.35)
    ep_r = 0
    if agent.args.mode == 'test':
        agent.load()
        state = env.reset()
        action_history = []
        reward_history = []
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(np.float32(action))
            action_history.append(action)
            reward_history.append(reward)
            if t>=50:
                idx = np.argmax(reward_history)
                action = action_history[idx]
                break
            state = next_state
        x,y = action
        img = env.generate(env.input,x,y)
        FAR,BCR,new_img,contours,ids,heights = env.calculate(img)
        r = env.createmodel(contours,ids,heights)

    elif agent.args.mode == 'train':
        total_step = 0
        state = env.reset()
        action_history = []
        reward_history = []
        for t in count():
            action = agent.select_action(state)
            action = (action + np.random.normal(0, agent.args.exploration_noise, size=action.shape)).clip(
                    -agent.args.max_action, agent.args.max_action)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))
            action_history.append(action)
            reward_history.append(reward)
            if done:
                break
            if t>=50:
                idx = np.argmax(reward_history)
                action = action_history[idx]
                break
            state = next_state
            agent.update()
        x,y = action
        img = env.generate(env.input,x,y)
        FAR,BCR,new_img,contours,ids,heights = env.calculate(img)
        r = env.createmodel(contours,ids,heights)
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()
