# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

bin_size = 10
# Possible interval values for monkey
bot_vals = np.arange(-155, 50, bin_size)
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.last_last_state = None
        self.last_last_action = None
        self.last_last_reward = None



        self.gravity = 0
        self.explorationCoef = .001
        self.eta = 0.99
        self.gamma = 0.99
        self.qs = {}

    def reset(self):
        # do death update:
        ourLastVel = self.last_last_state['monkey']['vel']
        last_bot_bot = self.last_last_state['monkey']['bot'] - self.last_last_state['tree']['bot']
        last_state_y = find_nearest(bot_vals, last_bot_bot)

        self.previousReward = 1 if self.last_last_state['tree']['dist']==460 else 0

        # current state: relHeight bin, x-dist, grav, velocity
        previousState = (last_state_y, self.last_last_state['tree']['dist'], self.gravity, ourLastVel)
        w = self.qs[(previousState, self.last_last_action)] - self.eta * (
                            self.qs[(previousState, self.last_last_action)] -
                            (self.previousReward + self.gamma * self.last_reward))
        self.qs[(previousState, self.last_last_action)] = w



        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):


        #print(self.gravity)
        starting = False
        if state['tree']['dist'] == 485:
            self.gravity = 0
            if self.last_reward == None:
                starting = True
        elif self.gravity == 0:
            self.gravity = state['monkey']['vel']

        if state['monkey']['bot'] < - state['monkey']['vel'] - self.gravity:
            return 0



        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        if not starting:
            # You might do some learning here based on the current state and the last state.
            ourVel = state['monkey']['vel']
            bot_bot = state['monkey']['bot'] - state['tree']['bot']
            state_y = find_nearest(bot_vals, bot_bot)

            ourLastVel = self.last_state['monkey']['vel']
            last_bot_bot = self.last_state['monkey']['bot'] - self.last_state['tree']['bot']
            last_state_y = find_nearest(bot_vals, last_bot_bot)

            #current state: relHeight bin, x-dist, grav, velocity
            currentState = (state_y, state['tree']['dist'], self.gravity, ourVel)
            previousState = (last_state_y, self.last_state['tree']['dist'], self.gravity, ourLastVel)

            for i in (0,1):
                if (currentState, i) not in self.qs:
                    self.qs[(currentState, i)] = 0

            maxQCurrent = max(self.qs[(currentState, 0)],self.qs[(currentState, 1)])
            bestOption = [self.qs[(currentState, 0)], self.qs[(currentState, 1)]].index(maxQCurrent)


            if (previousState, self.last_action) in self.qs:
                w = self.qs[(previousState, self.last_action)] - self.eta*(self.qs[(previousState, self.last_action)] -
                                                                     (self.last_reward + self.gamma*maxQCurrent))

                self.qs[(previousState, self.last_action)] = w
            else:
                self.qs[(previousState, self.last_action)] = 0


            # You'll need to select and action and return it.
            # Return 0 to swing and 1 to jump.


            randVal = npr.random()
            new_action = bestOption if randVal>self.explorationCoef else npr.randint(0,2)
        else:
            new_action = 0
        new_state  = state


        self.last_action = new_action
        self.last_state  = new_state
        self.last_last_state = state
        self.last_last_action = new_action

        return new_action
        #return 1 if state['monkey']['bot'] < state['tree']['bot'] - state['monkey']['vel'] - self.gravity else 0

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward



def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 2000, 50)

	# Save history. 
	np.save('hist',np.array(hist))


