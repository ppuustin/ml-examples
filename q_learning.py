import os, time

import matplotlib.pyplot as plt

import numpy as np

class Env:
    '''
    This is an Environment object, which roughly 
    implements Gym API like Environment https://gymnasium.farama.org/
    '''
    def __init__(self):
        self.height = 5
        self.width = 5
        self.posX = 0
        self.posY = 0
        self.endX = self.width-1
        self.endY = self.height-1
        self.actions = [0, 1, 2, 3]
        self.stateCount = self.height*self.width
        self.actionCount = len(self.actions)

    def reset(self):
        self.posX = 0
        self.posY = 0
        self.done = False
        return 0, 0, False

    def step(self, action):                                                      # take action
        if action == 0:                                                          # left
            self.posX = self.posX-1 if self.posX > 0 else self.posX
        if action == 1:                                                          # right
            self.posX = self.posX+1 if self.posX < self.width - 1 else self.posX
        if action == 2:                                                          # up
            self.posY = self.posY-1 if self.posY > 0 else self.posY
        if action == 3:                                                          # down
            self.posY = self.posY+1 if self.posY < self.height - 1 else self.posY

        done = self.posX == self.endX and self.posY == self.endY
        nextState = self.width * self.posY + self.posX                           # mapping (x,y) position to number between 0 and 5x5-1=24
        reward = 1 if done else 0
        return nextState, reward, done

    def randomAction(self):                                                      # return a random action
        return np.random.choice(self.actions)

    def render(self):                                                            # display environment
        for i in range(self.height):
            for j in range(self.width):
                if self.posY == i and self.posX == j:
                    print("O ", end='')
                elif self.endY == i and self.endX == j:
                    print("T ", end='')
                else:
                    print(". ", end='')
            print("")
            
# -----------------------------------------------------------------------------------

def train(env, qtable, epochs, gamma, epsilon, decay, step_sleep=0.1, done_sleep=0.3):

    scores, rewards = [], []    
    for i in range(epochs):                                                      # training loop
        state, reward, done = env.reset()
        steps, rewards_ = 0, 0
        while not done:
            os.system('cls') #os.system('clear')

            print("epoch #", i+1, "/", epochs)
            env.render()
            #print(np.round(qtable, 2))
            #time.sleep(0.3)                                                     # <-- sleep more 
            steps += 1                                                           # count steps to finish game
    
            if np.random.uniform() < epsilon:                                    # act randomly sometimes to allow exploration
                action = env.randomAction()
            else:                                                                # if not select max action in Qtable (act greedy)
                action = qtable[state].index(max(qtable[state]))
    
            next_state, reward, done = env.step(action)                          # take action
            qtable[state][action] = reward + gamma * max(qtable[next_state])     # update qtable value / Bellman equation
            
            #Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)] 
            #Q[(s,a)] += alpha * (r + gamma * Q[(s_,a_)]-Q[(s,a)])               # SARSA

            time.sleep(step_sleep)                                               # <-- sleep more     
            state = next_state                                                   # update state
            rewards_ += reward
     
        rewards.append(rewards_)
        epsilon -= decay*epsilon                                                 # The more we learn, the less we take random actions
    
        print('\nDone in', steps, 'steps'.format(steps))
        time.sleep(done_sleep)                                                   # <-- sleep more
    
        if (np.max(qtable) > 0):
            score = np.sum( qtable/np.max(qtable) * 100 )
            scores.append(score)

    mean_rewards = [np.mean(rewards[n-10:n]) if n > 10 else np.mean(rewards[:n]) # Calculate rolling average
                   for n in range(1, len(rewards))]

    return scores, rewards, mean_rewards

def plot(scores, rewards, mean_rewards, fname=None):
    fig, axs = plt.subplots(2)
    fig.suptitle('scores/rewards')
    axs[0].plot(scores, 'g')
    axs[0].legend(['scores'])
    axs[1].plot(rewards, 'gray')
    axs[1].plot(mean_rewards, 'orange')
    #plt.ylabel('Amount')
    plt.xlabel('episode')
    plt.legend(['rewards', 'rewards_avg'])
    axs[0].grid()
    axs[1].grid()
    plt.tight_layout()
    if fname is not None: plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()  


def main():
    env = Env()                                                        # create environment
    qtable = np.random.rand(env.stateCount, env.actionCount).tolist()  # QTable : contains the Q-Values for every (state,action) pair  
    epochs = 10                                                        # hyperparameters
    gamma = 0.1                                                        # discount factor
    epsilon = 0.08                                                     # amount of exploration
    decay = 0.1                                                        # decay of exploration
    
    scores, rewards, mean_rewards = train(env, qtable, epochs, gamma, epsilon, decay) # step_sleep=0.2, done_sleep=0.8
    plot(scores, rewards, mean_rewards)

if __name__ == '__main__':    
    start = time.time()
    main()
    print('DONE in : {0} seconds.'.format(time.time() - start))
