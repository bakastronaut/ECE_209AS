#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:00:43 2018

@author: johndoe
"""

import matplotlib as mpl
import time
from BigBang import BigBang,Agent,plotStateHistory
import matplotlib as mpl

#%%
World = BigBang(6,6,0,0.9)

IDs_red = [0,1,2,3,4,5,6,11,12,17,18,23,24,29,30,31,32,33,34,35]
rewards_red = [-100 for i in range(len(IDs_red))]

IDs_yellow = [8,10,14,16,20,22]
rewards_yellow = [-10 for i in range(len(IDs_yellow))]

IDs_green = [9]
rewards_green = [1]

World.AssignRewards([IDs_red,rewards_red])
World.AssignRewards([IDs_yellow,rewards_yellow])
World.AssignRewards([IDs_green,rewards_green])

World.InitializePolicy(9)

#%%
# 3(c)
agent0 = Agent(World,state=[6,7],patience=18)
agent0.Navigate()
plotStateHistory(World,agent0.statehist,World.rewardspace,'3(c) State History - Initial Poicy')
mpl.pyplot.savefig('3c.png',dpi=300,format='png')

#%%
# 3(e)
InitialPolicyValue = World.Value(World.policy,0.9)
value = 0
for state in agent0.statehist:
    theta = state[0]
    ID = state[1]
    value += InitialPolicyValue[theta,ID]

outputmsg = ' '.join(['Value of trajectory in 3(c):',str(value)])
print(outputmsg)

#%% POLICY ITERATION
# 3(h)
#t = time.time()
#agent1 = Agent(World,state=[6,7],patience=18)
#valuehist1 = World.PolicyIteration()
#mpl.pyplot.figure()
#mpl.pyplot.plot(valuehist1)
#mpl.pyplot.title('Policy Iteration')
#plotStateHistory(World,agent1.statehist,World.rewardspace,'Optimal State History - Policy Iteration')
#mpl.pyplot.savefig('3h.png',dpi=300,format='png')
#
## 3(i)
#elapsed_3i = time.time() - t

#%%
# 4(b)
t = time.time()
valuehist2 = World.ValueIteration()
agent2 = Agent(World,state=[6,7],patience=18)
agent2.Navigate()
mpl.pyplot.figure()
mpl.pyplot.plot(valuehist2)
mpl.pyplot.title('Value Iteration')
plotStateHistory(World,agent2.statehist,World.rewardspace,'Optimal State History - Value Iteration')
mpl.pyplot.savefig('4b.png',dpi=300,format='png')

# 4(c)
elapsed_4c = time.time() - t

#%%
# 5(a)
World2 = BigBang(6,6,0.25,0.9)

IDs_red = [0,1,2,3,4,5,6,11,12,17,18,23,24,29,30,31,32,33,34,35]
rewards_red = [-100 for i in range(len(IDs_red))]

IDs_yellow = [8,10,14,16,20,22]
rewards_yellow = [-10 for i in range(len(IDs_yellow))]

IDs_green = [9]
rewards_green = [1]

World2.AssignRewards([IDs_red,rewards_red])
World2.AssignRewards([IDs_yellow,rewards_yellow])
World2.AssignRewards([IDs_green,rewards_green])

World2.InitializePolicy(9)

#%%
valuehist3 = []
valuehist3 = World.ValueIteration()
agent3 = Agent(World,state=[6,7],patience=18)
agent3.Navigate()

OptimalPolicyValue = World2.Value(World.policy,0.9)
value = 0
for state in agent0.statehist:
    theta = state[0]
    ID = state[1]
    value += OptimalPolicyValue[theta,ID]

outputmsg = ' '.join(['Value of trajectory in 5(a):',str(value)])
print(outputmsg)
mpl.pyplot.figure()
mpl.pyplot.plot(valuehist3)
mpl.pyplot.title('Policy Iteration')
plotStateHistory(World,agent3.statehist,World.rewardspace,'5(a) Optimal State History - Value Iteration - Pe = 0.25')
mpl.pyplot.savefig('5a.png',dpi=300,format='png')