#Fareha Sultan 100968491
import numpy
import sys

class td_qlearning:

  alpha = 0.2
  gamma = 0.9
  qvalues={}

  def __init__(self, trial_filepath):
    trial =[]
    # trial_filepath is the path to a file containing a trial through state space
    squares = ['W','X','Y','Z']
    # A grid(x,y)  of squares ['W','X','Y','Z'] where Y=(0,0), Z=(0,1), W=(1,0), X=(1,1)
    with open(trial_filepath) as f: 
      for line in f.readlines():
        trial.append(line[:-1].split(','))
        f.close()

    for M in squares:
      for C  in squares:
        state = str(M+C) #state are represented as MC, where M=mouse's square and C=cat's square
        if (state[0] == 'W'): 
            # actions: up (U), down (D), left(L), right(R) and do not move (N)
            td_qlearning.qvalues[state] = {"D": 0, "R": 0, "N": 0} 
        if (state[0] == 'X'):
            td_qlearning.qvalues[state] = {"D": 0, "L": 0, "N": 0}
        if (state[0] == 'Y'):
            td_qlearning.qvalues[state] = {"U": 0, "R": 0, "N": 0}
        if (state[0] == 'Z'):
            td_qlearning.qvalues[state] = {"U": 0, "L": 0, "N": 0}
    
    for i in range(len(trial)-1):
        #Q(s,a)
        state = trial[i][0]
        action = trial[i][1]
        q_value = td_qlearning.qvalues[state][action] 
        #reward
        reward = td_qlearning.reward(state)
        #Q(s',a')
        state_prime = trial[i+1][0]
        action_prime = trial[i+1][1]
        q_prime_values = td_qlearning.qvalues[state_prime].values()
        #calling calculate 
        updated_qvalue = td_qlearning.calculate(q_value,reward,q_prime_values)
        #update the td_qlearning.qvalues
        td_qlearning.qvalues[state][action] = updated_qvalue
    
  def calculate(q,r,m):
      #The equation for Temporal Difference Q-learning
      #Q(s,a) <- Q(s,a) + alpha * ( r(s) + gamma * max_a'(Q(s',a')) - Q(s,a))
      newQ = q + td_qlearning.alpha *(r + td_qlearning.gamma * max(m) - q) 
      return newQ
           
  def reward(state):
      #Example: state = "WW"
      if(state[0]==state[1]):
          return -1
      else:
        return 1  
  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is a string representation of an action
    q = td_qlearning.qvalues[state][action]
    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state
    a = max(td_qlearning.qvalues[state], key = (td_qlearning.qvalues[state]).get)  
    # Return the optimal action under the learned policy
    return a
  
  
  

#td = td_qlearning("Examples/Examples/Example0/trial.csv")

