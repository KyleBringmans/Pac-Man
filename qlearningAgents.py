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



class NeuralQAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.parameters = {}
        self.losses = []

        self.initialized = False

    def initializeParameters(self):
        # Initializes the parameters W and b of the networl

        # dimensions of the network ex. (#input, #hidden1, #hidden2, #output)
        layerDims = self.layerDims

        # number of layers in the network
        L = len(layerDims)

        for l in range(1, L):
            self.parameters['W-%d' % l] = np.random.randn(layerDims[l], layerDims[l - 1]) * 0.01
            self.parameters['b-%d' % l] = np.zeros((layerDims[l], 1))

            assert (self.parameters['W-%d' % l].shape == (layerDims[l], layerDims[l - 1]))
            assert (self.parameters['b-%d' % l].shape == (layerDims[l], 1))

    def getFeatures(self, state, action):
        # Computes the feature-vector for this (state, action)

        # data for evaluating features
        # about the board
        food = state.getFood()
        capsules = state.getCapsules()
        walls = state.getWalls()
        crossroads = state.getCrossroads()
        numGhosts = state.getNumGhosts()

        # about the ghosts
        ghostsPos = state.getGhostPositions()
        ghostsStates = state.getGhostStates()
        ghostsScared = 0
        for i in range(0, len(ghostsStates)): ghostsScared += int(ghostsStates[i].isScared())
        ghostsScared = (ghostsScared > 0)

        # about Pacman
        pacmanPos = state.getPacmanPosition()
        pacmanNextPos = vectorSum(pacmanPos, Actions.directionToVector(action))
        pacmanNextPos = (int(pacmanNextPos[0]), int(pacmanNextPos[1]))
        pacmanNextLegal = state.generatePacmanSuccessor(action).getLegalPacmanActions()
        pacmanLastDirection = state.data.agentStates[0].getDirection()
        pacmanNextCrossroad = chooseNextCrossroad(crossroads, ghostsPos, pacmanNextPos, self.distMap, walls)

        # parameters for normalizing
        maxFood = (food.height * food.width)
        maxDist = self.distMap[max(self.distMap, key=self.distMap.get)]
        maxActions = 5

        # feature-vector
        features = {}
        features["bias"] = 1

        # features related to the board
        features["progress"] = float(state.getNumFood()) / maxFood
        features["numGhosts"] = numGhosts

        # features related to pacman
        features["PM-num-legal-actions"] = float(len(pacmanNextLegal)) / maxActions
        features["PM-dist-food"] = float(maxDist - closestFood(pacmanNextPos, food, walls)) / maxDist
        if capsules:
            features["PM-dist-capsule"] = float(
                maxDist - distToClosestCapsule(pacmanNextPos, capsules, self.distMap)) / maxDist
        features["PM-dist-NextCrossroad"] = float(maxDist - self.distMap[pacmanNextPos, pacmanNextCrossroad]) / maxDist
        features["PM-dist-closest-edible-ghost"] = max(0, int(ghostsScared))

        # features related to ghosts
        for i in range(0, numGhosts):
            ghostIsScared = ghostsStates[i].isScared()
            scaredTimer = ghostsStates[i].getScaredTimer()
            ghostPos = ghostsStates[i].getPosition()
            ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
            ghostsStartPos = ghostsStates[i].start.getPosition()
            distPmGh = self.distMap[pacmanNextPos, ghostPos]

            features["GH-%d-scaredTimer" % i] = float(scaredTimer) / pacman.SCARED_TIME
            features["GH-%d-dist-to-initial-pos" % i] = float(self.distMap[ghostsStartPos, ghostPos]) / maxDist
            features["GH-%d-dist-PM" % i] = float(self.distMap[pacmanNextPos, ghostPos]) / maxDist
            features["GH-%d-dist-NextCrossroad" % i] = float(
                maxDist - self.distMap[pacmanNextPos, pacmanNextCrossroad]) / maxDist
            # features["GH-%d-approaching" % i] =
            # features["GH-%d-speed" % i] =

            if ghostIsScared and (float(distPmGh) / maxDist) < features["PM-dist-closest-edible-ghost"]:
                features["PM-dist-closest-edible-ghost"] = float(distPmGh) / maxDist

        if not self.initialized:
            self.numFeatures = len(features)

            self.layerDims = (self.numFeatures, 10, 4, 1)
            self.initializeParameters()
            self.initialized = True

        F = np.array([features[feature] for feature in features])
        F = F.reshape(self.numFeatures, 1)

        return F

    def linearForward(A, W, b):
        # Computes the linear part of a layer's forward propagation

        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    linearForward = staticmethod(linearForward)

    def linearActivationForward(self, A_prev, W, b, activation):
        # Computes the forward propagation for the linear -> activation layer

        if activation == "sigmoid":
            Z, linear_cache = self.linearForward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linearForward(A_prev, W, b)
            A, activation_cache = relu(Z)

        elif activation == "tanh":
            Z, linear_cache = self.linearForward(A_prev, W, b)
            A, activation_cache = tanh(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def modelForward(self, X):
        # Computes the output-value of the network, given the input-vector

        caches = []
        A = X
        L = len(self.layerDims)  # number of layers in the network

        for l in range(1, L):
            A_prev = A
            A, cache = self.linearActivationForward(A_prev, self.parameters['W-%d' % l], self.parameters['b-%d' % l],
                                                    'relu')
            caches.append(cache)

        AL, cache = self.linearActivationForward(A, self.parameters['W-%d' % L], self.parameters['b-%d' % L], 'tanh')
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def linearBackward(dZ, cache):
        # Computes the linear part of a layer's backward propagation

        A_prev, W, b = cache
        m = A_prev.shape[1]  # number of training examples

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    linearBackward = staticmethod(linearBackward)

    def linearActivationBackward(self, dA, cache, activation):
        # Computes the backward propagation for the linear -> activation layer

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = reluBackward(dA, activation_cache)
            dA_prev, dW, db = self.linearBackward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoidBackward(dA, activation_cache)
            dA_prev, dW, db = self.linearBackward(dZ, linear_cache)

        elif activation == "tanh":
            dZ = tanhBackward(dA, activation_cache)
            dA_prev, dW, db = self.linearBackward(dZ, linear_cache)

        return dA_prev, dW, db

    def modelBackward(self, AL, Y, caches):
        # Computes the gradients for updating the network's parameters
        # dA = dL/dA, dW = dL/dW, db = dL/db with L the loss function

        grads = {}
        L = len(self.layerDims)  # the number of layers
        # m = AL.shape[1]
        # Y = Y.reshape(AL.shape)

        # Initializing the backward propagation
        # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dAL = -2 * AL

        # Lth layer (TANH -> LINEAR) gradients
        current_cache = caches[L - 1]
        grads['dA-%d' % L], grads['dW-%d' % L], grads['db-%d' % L] = self.linearActivationBackward(dAL, current_cache,
                                                                                                   'tanh')

        # Loop from L-2 to 0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linearActivationBackward(grads['dA-%d' % (l + 1)], current_cache,
                                                                           'relu')
            grads['dA-%d' % l] = dA_prev_temp
            grads['dW-%d' % (l + 1)] = dW_temp
            grads['db-%d' % (l + 1)] = db_temp

        return grads

    def getQValue(self, state, action):
        # Returns the q-value of this (state, action)

        features = self.getFeatures(state, action)
        qValue, cache = self.modelForward(features)

        return qValue

    def computeValueFromQValues(self, state):
        # Returns max_action Q(state,action)

        legal = state.getLegalActions(self.index)
        qList = []  # store the q-values of a state

        # if no legal actions, it is a terminal state, return 0
        if not legal:
            qList.append(0)
        # otherwise store the q-values and return the highest one
        else:
            for action in legal:
                qList.append(self.getQValue(state, action))

        return max(qList)

    def computeActionFromQValues(self, state):
        # Computes the best action to take in a state.

        legal = state.getLegalActions(self.index)

        # loop over the possible actions and choose the one with the highest q-value
        bestAction = random.choice(legal)
        for action in legal:
            if self.getQValue(state, action) > self.getQValue(state, bestAction):
                bestAction = action

        return bestAction

    def getAction(self, state):
        # Returns the action to take in a state

        legal = state.getLegalActions(self.index)  # store the possible actions
        action = random.choice(legal)  # the final action the agent undertakes

        if self.isInTraining() and self.manualTrain:
            action = self.getManualMove(state)
        else:
            # if there are legal actions, choose the best action only sometimes, for allowing some exploration
            if legal:
                if util.flipCoin(self.epsilon):
                    action = random.choice(legal)
                else:
                    action = self.computeActionFromQValues(state)

        self.doAction(state, action)
        return action

    def updateParameters(self, grads):
        # Updates the parameters of the network

        L = len(self.layerDims)  # number of layers in the network

        for l in range(L):
            self.parameters['W-%d' % (l + 1)] -= self.alpha * grads['dW-%d' % (l + 1)]
            self.parameters['b-%d' % (l + 1)] -= self.alpha * grads['db-%d' % (l + 1)]

        self.epsilon = max(self.epsilon - 0.01, self.epsilon_end)

    def update(self, state, action, nextState, reward):
        # Performs one forward and one backward propagation

        X = self.getFeatures(state, action)

        # Forward propagation
        AL, caches = self.modelForward(X)

        # Normalize reward
        reward = float(reward) / pacman.MAX_REWARD
        # Compute cost
        loss = self.computeLoss(AL, reward, state, action, nextState)
        self.losses.append(loss)

        # Backward propagation
        grads = self.modelBackward(AL, reward, caches)

        # Update parameters
        self.updateParameters(grads)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.parameters

    def setWeights(self, params):
        self.parameters = params

    def computeLoss(self, AL, reward, state, action, nextState):
        # Computes the error of the prediction
        # AL is the output of the output layer

        # cost = - np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        y = reward + self.discount * self.computeValueFromQValues(nextState)
        loss = (y - AL) ** 2  # missing V(y)

        loss = np.squeeze(loss)
        assert (loss.shape == ())

        return loss

    def final(self, state):
        ReinforcementAgent.final(self, state)

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
        self.weightsNScared = util.Counter()

    def getWeightsNScared(self):
        return self.weightsNScared

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state,action)

        q = 0
        for feat in features:   # Q(s,a) = Sum: i -> n : f_i(s,a)*w_i
            q += features[feat] * self.getWeightsNScared()[feat] # formula summed over
        return q

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state,action)
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state,action) # seperate because it doesn't work otherwise :///
        for feat in features:
            self.weightsNScared[feat] += self.alpha * difference * features[feat] # w_i + alfa * difference * f_i(s,a)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
