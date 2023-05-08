# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            outValues = util.Counter()
            for eachState in self.mdp.getStates():
                tempValues = util.Counter()
                for eachAction in self.mdp.getPossibleActions(eachState):
                    tempValues[eachAction] = self.getQValue(eachState, eachAction)

                outValues[eachState] = tempValues[tempValues.argMax()]
            self.values = outValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tProbability = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0.0
        for transS, transP in tProbability:
            qValue += transP * (self.mdp.getReward(state, action, transS) \
                               +self.discount*self.getValue(transS))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:
            qValue = util.Counter()
            for eachAction in self.mdp.getPossibleActions(state):
                qValue[eachAction] = self.computeQValueFromValues(state, eachAction)
            return qValue.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        for i in range(self.iterations):
            modState = allStates[i %  len(self.mdp.getStates())]
            compAction = self.computeActionFromValues(modState)
            if compAction:
                qFromV = self.computeQValueFromValues(modState, compAction)
            else:
                qFromV = 0.0
            self.values[modState] = qFromV

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        predecessors = {}
        for s in allStates:
            predecessors[s] = set()
        for eachS in allStates:
            for eachA in self.mdp.getPossibleActions(eachS):
                for transState, prob in self.mdp.getTransitionStatesAndProbs(eachS, eachA):
                    #print(prob)
                    if prob > 0.0:
                        predecessors[transState].add(eachS)

        prioQue = util.PriorityQueue()
        #print(prioQue.isEmpty())

        for eachS in allStates:
            if self.mdp.isTerminal(eachS) == False:
                qValue = []
                for eachA in self.mdp.getPossibleActions(eachS):
                    qValue.append(self.getQValue(eachS, eachA))
                dq = abs(self.getValue(eachS) - max(qValue))
                # push(self, item, priority)
                prioQue.push(eachS, -dq)

        for i in range(self.iterations):
            if prioQue.isEmpty() == False:
                nowState = prioQue.pop()
                qValue = []
                if self.mdp.isTerminal(nowState) == False:
                    for eachA in self.mdp.getPossibleActions(nowState):
                        qValue.append(self.getQValue(nowState, eachA))

                    self.values[nowState] = max(qValue)
                for peachPDS in predecessors[nowState]:
                    qValue = []
                    if self.mdp.isTerminal(peachPDS) == False:
                        for action in self.mdp.getPossibleActions(peachPDS):
                            qValue.append(self.getQValue(peachPDS, action))
                        dqp = abs(self.getValue(peachPDS) - max(qValue))
                    else:
                        dqp = abs(self.getValue(peachPDS))
                    
                    if dqp > self.theta:
                        prioQue.update(peachPDS, -dqp)
            else:
                return