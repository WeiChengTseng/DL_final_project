from collections import namedtuple
import random
import numpy as np 
# Experience = namedtuple('Experience',
#                         ('states', 'next_state', 'actions', 'rewards'))


# class ReplayMemory:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Experience(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)
#----------------------------------------------------#

######################################################
            #########################  # 1 data 
            #          #2           #
            #     #0                # blue team
            #                       #
            #                       #
            #                       #
            #                #1     #
            #                       # red team
            #          #3           #
            #########################
######################################################

class ReplayBuffer:
    def __init__(self,field=8, gamma=0.95,capacity=1e6):
        self.field = field
        self._trags = [Trajetory(2,2) for i in range(field)]
        self._record = np.zeros(field, dtype=bool)
        self.actions = [[]]*4
        self.states = [[]]*4
        self.probs  = [[]]*4
        self.rewards = [[]]*4
        self.next_states = [[]]*4
        self.capacity = capacity
        return 

    def sample(self, order, seed, batchSize):
        batch_states = [[]]*4
        batch_actions= [[]]*4
        batch_probs = [[]]*4
        batch_rewards = [[]]*4
        batch_next_states = [[]] *4
        # print("memory",len(self.states[0]))
        for i in order:
            random.seed(seed)
            batch_states[i]=random.choices(self.states[i],k=batchSize)
            random.seed(seed)
            batch_actions[i]=random.choices(self.actions[i],k=batchSize)
            random.seed(seed)
            batch_probs[i]=random.choices(self.probs[i],k=batchSize)
            random.seed(seed)
            batch_rewards[i]=random.choices(self.rewards[i],k=batchSize)
            random.seed(seed)
            batch_next_states[i]=random.choices(self.next_states[i],k=batchSize)
        # print("batch",len(batch_states[0]))
        return batch_states, batch_actions, batch_probs, batch_rewards, batch_next_states

    def clear_memory(self):
        if len(self.actions[0]) > self.capacity:
            for i in range(4):
                del self.actions[i][:-capacity]
                del self.states[i][:-capacity]
                del self.probs[i][:-capacity]
                del self.rewards[i][:-capacity]
                del self.next_states[i][:-capacity]
        return
    
    
    def update_transition(self, state_s, state_g, action_s,action_g, prob_s,prob_g, reward_s,reward_g, done,next_state_s,next_state_g):
        # not modify
        for i in range(0,self.field):
            if i in np.argwhere(done == False).flatten():
                self._trags[i].push_transition('str',state_s[i*2:(i+1)*2], action_s[i*2:(i+1)*2],prob_s[i*2:(i+1)*2], reward_s[i*2:(i+1)*2], next_state_s[i*2:(i+1)*2])
                self._trags[i].push_transition('goalie',state_g[i*2:(i+1)*2], action_g[i*2:(i+1)*2],prob_g[i*2:(i+1)*2], reward_g[i*2:(i+1)*2], next_state_g[i*2:(i+1)*2])
        return
    
    def update_rewards(self,done):
        for i in (np.argwhere(done == True).flatten()):
            if i is []:
                break
            # print("self_record", self._record)
            if i in np.argwhere(self._record == False).flatten():
                # print(i)
                s, p, a, r ,ns = self._trags[i].done()
                # print("len_s",len(s[0]))
                
                for j in range(4):
                    if len(self.states[j]) == 0:
                        self.states[j] = list(s[j])
                        self.probs[j] = list(p[j])
                        self.actions[j] = list(a[j])
                        self.rewards[j] = list(r[j])
                        self.next_states[j] = list(ns[j])
                    else:
                        self.states[j] += list(s[j])
                        self.probs[j] += list(p[j])
                        self.actions[j] += list(a[j])
                        self.rewards[j] += list(r[j])
                        self.next_states[j] += list(ns[j])
                    
                self._trags[i].clear()
                self._record[i] = True
        return
    


class Trajetory:
    def __init__(self,red_actor=2,blue_actor=2):
        self.red_actor  = red_actor
        self.blue_actor = blue_actor
        self._next_state = [[]]*(int(red_actor)+int(blue_actor))
        self._state = [[]]*(int(red_actor)+int(blue_actor))
        self._action = [[]]*(int(red_actor)+int(blue_actor))
        self._prob = [[]]*(int(red_actor)+int(blue_actor))
        self._reward = [[]]*(int(red_actor)+int(blue_actor))
        self.record = False
        
        return

    def push_transition(self,mode, state, action, prob, reward, next_state):
        if mode == "str":
            if len(self._state[0]) == 0:
                self._state[0] = [state[0]]
                self._action[0] = [action[0]]
                self._prob[0] = [prob[0]]
                self._reward[0] = [reward[0]]
                self._next_state[0] = [next_state[0]]
                self._state[1] = [state[1]] 
                self._action[1] = [action[1]]
                self._prob[1] = [prob[1]]
                self._reward[1] = [reward[1]]
                self._next_state[1] = [next_state[1]]
            else:
                self._state[0].append(state[0])
                self._action[0].append(action[0])
                self._prob[0].append(prob[0])
                self._reward[0].append(reward[0])
                self._next_state[0].append(next_state[0])
                self._state[1].append(state[1])
                self._action[1].append(action[1])
                self._prob[1].append(prob[1])
                self._reward[1].append(reward[1])
                self._next_state[1].append(next_state[1])
        if mode == "goalie":
            if len(self._state[2]) == 0:
                self._state[2] = [state[0]]
                self._action[2] = [action[0]]
                self._prob[2] = [prob[0]]
                self._reward[2] = [reward[0]]
                self._next_state[2] = [next_state[0]]
                self._state[3] = [state[1]] 
                self._action[3] = [action[1]]
                self._prob[3] = [prob[1]]
                self._reward[3] = [reward[1]]
                self._next_state[3] = [next_state[1]]
            else:
                self._state[2].append(state[0])
                self._action[2].append(action[0])
                self._prob[2].append(prob[0])
                self._reward[2].append(reward[0])
                self._next_state[2].append(next_state[0])
                self._state[3].append(state[1])
                self._action[3].append(action[1])
                self._prob[3].append(prob[1])
                self._reward[3].append(reward[1])
                self._next_state[3].append(next_state[1])
        return

    def done(self):
        return self._state, self._prob, self._action, self._reward, self._next_state

    def clear(self):
        for i in range(self.red_actor+self.blue_actor):
            del self._action[i][:]
            del self._state[i][:]
            del self._prob[i][:]
            del self._reward[i][:]
        return
        
        

    


    

        