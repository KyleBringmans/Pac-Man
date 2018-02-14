# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    # if there is no noise, then since future rewards are barely discounted the agent will cross the bridge
    return answerDiscount, answerNoise

# ----------------------------------------------------------------------------------------------------------------------
# these values were eyeballed and weren't outright calculated

def question3a():
    answerDiscount = 0.1 # heavily discount distant rewards -> not worth taking
    answerNoise = 0 # remove noise to make risking cliff feasable
    answerLivingReward = -1 # make the agent want to stop at an end-state
    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    answerDiscount = 0.5
    answerNoise = 0.4 # add noise to make falling of cliff a larger possibility
    answerLivingReward = -1 # make the agent want to stop at an end-state
    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    answerDiscount = 0.9 # distant rewards are barely discounted
    answerNoise = 0  # no risk of falling off cliff
    answerLivingReward = -1 # make the agent want to stop at an end-state
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.9 # distant rewards are more attractive
    answerNoise = 0.4 # cliff is too risky because of noise
    answerLivingReward = -1 # make the agent want to stop at an end-state
    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    answerDiscount = 0.5 # eyeballed this
    answerNoise = 0.4 # eyeballed this as well
    answerLivingReward = 5 # make ending extremely unattractive
    return answerDiscount, answerNoise, answerLivingReward


def question6():
    # answerEpsilon = 0
    # answerLearningRate = 0.1
    return 'NOT POSSIBLE'


if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
