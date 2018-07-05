# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from learningAgents import ReinforcementAgent
from featureExtractors import *
import pacman

import random,math
from util import sigmoid, relu, tanh, sigmoidBackward, reluBackward, tanhBackward

import numpy as np


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = util.Counter() # All q-values set to 0


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        if (state,action) not in self.qValues:
            self.qValues[(state,action)] = 0.0  # unseen state
        return self.qValues[(state,action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if len(self.getLegalActions(state)) == 0: # check for terminal state
            return 0.0
        tempVals = util.Counter()
        for action in self.getLegalActions(state):  # calculate q-values for every possible action
            tempVals[(action)] = self.getQValue(state,action)
        return tempVals[tempVals.argMax()]
        # max_a_Q(s_t+1,a)

#TODO!!!
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        actionVals = self.getLegalActions(state)

        if len(actionVals) == 0:
            return None


        tempVals = util.Counter() # store q-values in here


        for action in actionVals:
            tempVals[action] = self.getQValue(state, action)
        return tempVals.argMax() # takes the random choice



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        self.qValues[(state,action)] = (1-self.alpha) * self.getQValue(state,action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
        # Q(s,a) = (1-a) * Q(s,a) + a * (R + gamma * max_a_Q(s_t+1,a))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.weightsScared = util.Counter()

    def getWeights(self):
        return self.weights

    def getWeightsScared(self):
        return self.weightsScared

    #EDITTED
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state,action)

        #counter = 0
        #p = state.data.agentStates[0].configuration.pos
        #useOtherVector = False
        #for agentState in state.data.agentStates:
            #if agentState.scaredTimer > 0:
            #    counter += 1
        #    agentPos = agentState.getPosition()
        #    if agentState.scaredTimer > 0 and util.euclDist(agentPos[0], agentPos[1], p[0], p[1]) :
        #        useOtherVector = True
        #if counter == len(state.data.agentStates) - 1:
        #    useOtherVector = True

        useOtherVector = False
        p = state.data.agentStates[0].configuration.pos
        g1 = state.data.agentStates[1]
        g2 = state.data.agentStates[2]
        distg1 = util.euclDist(p[0], p[1], g1.configuration.pos[0],
                               g1.configuration.pos[1])
        distg2 = util.euclDist(p[0], p[1], g2.configuration.pos[0],
                               g2.configuration.pos[1])
        if (g1.scaredTimer > 0 and distg1 < distg2) or (g2.scaredTimer > 0 and distg2 < distg1):
            useOtherVector = True

        if useOtherVector:
            q = 0
            for feat in features:  # Q(s,a) = Sum: i -> n : f_i(s,a)*w_i
                q += features[feat] * self.getWeightsScared()[feat]  # formula summed over
            return q
        else:
            q = 0
            for feat in features:   # Q(s,a) = Sum: i -> n : f_i(s,a)*w_i
                q += features[feat] * self.getWeights()[feat]  # formula summed over
            return q

    #Editted
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state,action)
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state,action)  # seperate because it doesn't work otherwise :///

        useOtherVector = False
        p = state.data.agentStates[0].configuration.pos
        g1 = state.data.agentStates[1]
        g2 = state.data.agentStates[2]
        distg1 = util.euclDist(p[0], p[1], g1.configuration.pos[0], g1.configuration.pos[1])
        distg2 = util.euclDist(p[0], p[1], g2.configuration.pos[0], g2.configuration.pos[1])
        if (g1.scaredTimer > 0 and distg1 < distg2) or (g2.scaredTimer > 0 and distg2 < distg1):
            useOtherVector = True


        if useOtherVector:
            for feat in features:
                self.weightsScared[feat] += self.alpha * difference * features[feat]  # w_i + alfa * difference * f_i(s,a)
        else:
            for feat in features:
                self.weights[feat] += self.alpha * difference * features[feat]  # w_i + alfa * difference * f_i(s,a)

        #print(self.weights)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass