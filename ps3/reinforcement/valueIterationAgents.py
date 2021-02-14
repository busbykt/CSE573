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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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

        for iteration in range(self.iterations):
            nextvals = util.Counter()
            for state in self.mdp.getStates():
                action = self.computeActionFromValues(state)
                if action == None:
                    continue
                nextvals[state] = self.computeQValueFromValues(state, action)
            self.values=nextvals

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
        q = 0
        for nextState,prob in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, nextState)
            t=prob
            vsprime = self.values[nextState]
            q = q+t*(r+self.discount*vsprime)
        return q

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

        actions = self.mdp.getPossibleActions(state) # get possible Actions
        actiondict = util.Counter()

        for action in actions:
            q = self.computeQValueFromValues(state,action)
            actiondict[action] = q

        return actiondict.argMax()

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

        numstates = len(self.mdp.getStates())

        for iteration in range(self.iterations):
            state = self.mdp.getStates()[iteration%numstates]
            action = self.computeActionFromValues(state)
            if action == None:
                continue
            nextval = self.computeQValueFromValues(state, action)
            self.values[state]=nextval

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

        predecessor_dict={}

        for state in self.mdp.getStates():
            predecessor_dict[state]=set()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                ns_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
                for ns,prob in ns_prob_pairs:
                    predecessor_dict[ns].add(state)


        pQ = util.PriorityQueue()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            qlist=[]
            for action in self.mdp.getPossibleActions(state):
                qlist.append(self.computeQValueFromValues(state, action))

            maxQ = max(qlist) # fun aerospace reference https://en.wikipedia.org/wiki/Max_q
            diff = abs(self.values[state] - maxQ)
            pQ.update(state,-diff)

        for iteration in range(self.iterations):
            if pQ.isEmpty():
                return
            state = pQ.pop()
            if self.mdp.isTerminal(state):
                continue

            action = self.computeActionFromValues(state)
            if action == None:
                continue
            state_val = self.computeQValueFromValues(state, action)
            self.values[state] = state_val

            for p in predecessor_dict[state]:
                qlist=[]
                for action in self.mdp.getPossibleActions(p):
                    qlist.append(self.computeQValueFromValues(p, action))

                maxQ = max(qlist)
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    pQ.update(p,-diff)
