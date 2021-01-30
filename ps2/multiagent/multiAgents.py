# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [x.configuration.pos for x in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        curPos = currentGameState.getPacmanPosition()
        curFood = currentGameState.getFood()

        score=0

        newfoodDists=[]
        curfoodDists=[]
        # get current and new food distances
        for foodPos in newFood.asList():
            newfoodDists.append(l1Dist(newPos,foodPos))
            curfoodDists.append(l1Dist(curPos,foodPos))

        newfoodDists.sort()
        curfoodDists.sort()

        n = len(newfoodDists)

        # increase score if closer to remaining food weighted by distances
        if newfoodDists:
            if n>5 and sum(newfoodDists[:5]) < sum(curfoodDists[:5]):
                score = score+1
            if newfoodDists[0] < curfoodDists[0]:
                score = score+5

        x,y = newPos
        if curFood[x][y] == True:
            score = score+10

        # decrease score if close to ghost
        for newGhostPos in newGhostPositions:
            if sum(newScaredTimes) == 0:
                if l1Dist(newPos,newGhostPos)==2:
                    score = score-5
                if l1Dist(newPos,newGhostPos)==1:
                    score = score-100
                if l1Dist(newPos,newGhostPos)==0:
                    score = score-500
            elif sum(newScaredTimes) > 2:
                if l1Dist(newPos,newGhostPos)==2:
                    score = score+5
                if l1Dist(newPos,newGhostPos)==1:
                    score = score+10
                if l1Dist(newPos,newGhostPos)==0:
                    score = score+100

        # incentivize moving
        if curPos == newPos:
            score = score-1

        return score

def l1Dist(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.a=float('-inf')
        self.b=float('inf')


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        actions={}
        for action in gameState.getLegalActions(0):
            v = value(gameState.generateSuccessor(0,action), self.evaluationFunction, self.depth*gameState.getNumAgents(),0)
            actions[action] = v
        return max(actions, key=actions.get)



def value(state, evalfn,moves,agent):

    # keep track of the depth
    moves = moves-1

    # keep track of the agent
    agent = (agent +1)%state.getNumAgents()

    # if state is terminal state, return state utility
    if state.isWin() or state.isLose() or moves==0:
        return evalfn(state)

    # if next agent is max return max-value(state)
    if agent == 0:
        return maxValue(state,agent,evalfn,moves)
    # if next agent is min return min-value(state)
    if agent > 0:
        return minValue(state,agent,evalfn,moves)


def maxValue(state,agent,evalfn,moves):

    v=float('-inf')
    for action in state.getLegalActions(agent):
        successor = state.generateSuccessor(agent,action)
        v=max(v,value(successor,evalfn,moves,agent))
    return v

def minValue(state,agent,evalfn,moves):

    v=float('inf')
    for action in state.getLegalActions(agent):
        successor = state.generateSuccessor(agent,action)
        v=min(v,value(successor,evalfn,moves,agent))
    return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        actions={}
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,action)
            self.b=float('inf')
            v = self.abvalue(state=nextState,
                        evalfn=self.evaluationFunction,
                        moves=self.depth*gameState.getNumAgents(),
                        agent=0,
                        a=self.a,
                        b=self.b)
            self.a = max(self.a,v)
            #self.b = float('inf')
            actions[action] = v
        return max(actions, key=actions.get)

    def abvalue(self,state, evalfn,moves,agent,a,b):

        # keep track of the depth
        moves = moves-1

        # keep track of the agent
        agent = (agent+1)%state.getNumAgents()

        # if state is terminal state, return state utility
        if state.isWin() or state.isLose() or moves==0:
            return evalfn(state)

        # if next agent is max return max-value(state)
        if agent == 0:
            return self.abmaxValue(state,agent,evalfn,moves,self.a,self.b)
        # if next agent is min return min-value(state)
        if agent > 0:
            return self.abminValue(state,agent,evalfn,moves,self.a,self.b)


    def abmaxValue(self,state,agent,evalfn,moves,a,b):

        v=float('-inf')
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent,action)
            vn = self.abvalue(successor,evalfn,moves,agent,self.a,self.b)
            v=max(v,vn)

            print('agent',agent)
            print('v',v)
            print('b',b)
            print('a',a)
            input()

            if v>self.b: return v
            self.a=max(self.a,v)
        return v

    def abminValue(self,state,agent,evalfn,moves,a,b):

        v=float('inf')
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent,action)
            vn = self.abvalue(successor,evalfn,moves,agent,self.a,self.b)
            v=min(v,vn)

            print('agent',agent)
            print('v',v)
            print('b',b)
            print('a',a)
            input()

            if v<self.a: return v
            self.b=min(self.b,v)
        return v



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions={}
        for action in gameState.getLegalActions(0):
            v = self.evalue(gameState.generateSuccessor(0,action), self.evaluationFunction, self.depth*gameState.getNumAgents(),0)
            actions[action] = v
        return max(actions, key=actions.get)


    def evalue(self,state,evalfn,moves,agent):

        # keep track of the depth
        moves = moves-1

        # keep track of the agent
        agent = (agent +1)%state.getNumAgents()

        # if state is terminal state, return state utility
        if state.isWin() or state.isLose() or moves==0:
            return evalfn(state)

        # if next agent is max return max-value(state)
        if agent == 0:
            return self.emaxValue(state,agent,evalfn,moves)
        # if next agent is min return min-value(state)
        if agent > 0:
            return self.expValue(state,agent,evalfn,moves)

    def emaxValue(self, state, agent,evalfn,moves):
        v=float('-inf')
        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0,action)
            v=max(v,self.evalue(successor,evalfn,moves,agent))
        return v

    def expValue(self, state, agent,evalfn,moves):
        v=0
        numActions = len(state.getLegalActions(agent))
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent,action)
            v+=1/numActions*self.evalue(successor,evalfn,moves,agent)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    numFood = currentGameState.getNumFood()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [x.configuration.pos for x in ghostStates]
    ScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # feature 1 is 1 over the number of food pellets left
    w1=1
    if numFood < 1:
        numFood=.5
    f1 = 1/numFood

    # feature 2 is 1 over sum of l1 distances to all food
    w2=1
    curfoodDists=[]
    for foodPos in foodGrid.asList():
        curfoodDists.append(l1Dist(pacPos,foodPos))
    curfoodDists.sort()
    curfoodDist = sum(curfoodDists)
    if curfoodDist < 1:
        curfoodDist=.5
    if len(curfoodDists) == 0:
        w1=10
    f2=1/curfoodDist

    # feature 3 is distance to closest ghost
    w3=1

    ghostDists = [l1Dist(pacPos,ghostPos) for ghostPos in ghostPositions]
    ghostPos = [x for _,x in sorted(zip(ghostDists,ghostPositions))][0]

    f3=0
    if sum(ScaredTimes) > 2:
        if l1Dist(pacPos,ghostPos)>2:
            f3 = -1
        if l1Dist(pacPos,ghostPos)==2:
            f3 = 2
        if l1Dist(pacPos,ghostPos)==1:
            f3 = 10
        if l1Dist(pacPos,ghostPos)==0:
            f3 = 500
    else:
        if l1Dist(pacPos,ghostPos)>2:
            f3 = 1
        if l1Dist(pacPos,ghostPos)==2:
            f3 = -2
        if l1Dist(pacPos,ghostPos)==1:
            f3 = -10
        if l1Dist(pacPos,ghostPos)==0:
            f3 = -500

    return w1*f1+w2*f2+w3*f3

    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newGhostPositions = [x.configuration.pos for x in newGhostStates]
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #
    # "*** YOUR CODE HERE ***"
    # curPos = currentGameState.getPacmanPosition()
    # curFood = currentGameState.getFood()
    #
    # score=0
    #
    # newfoodDists=[]
    # curfoodDists=[]
    # # get current and new food distances
    # for foodPos in newFood.asList():
    #     newfoodDists.append(l1Dist(newPos,foodPos))
    #     curfoodDists.append(l1Dist(curPos,foodPos))
    #
    # newfoodDists.sort()
    # curfoodDists.sort()
    #
    # n = len(newfoodDists)
    #
    # # increase score if closer to remaining food weighted by distances
    # if newfoodDists:
    #     if n>5 and sum(newfoodDists[:5]) < sum(curfoodDists[:5]):
    #         score = score+1
    #     if newfoodDists[0] < curfoodDists[0]:
    #         score = score+5
    #
    # x,y = newPos
    # if curFood[x][y] == True:
    #     score = score+10
    #
    # # decrease score if close to ghost
    # for newGhostPos in newGhostPositions:
    #     if sum(newScaredTimes) == 0:
    #         if l1Dist(newPos,newGhostPos)==2:
    #             score = score-5
    #         if l1Dist(newPos,newGhostPos)==1:
    #             score = score-100
    #         if l1Dist(newPos,newGhostPos)==0:
    #             score = score-500
    #     elif sum(newScaredTimes) > 2:
    #         if l1Dist(newPos,newGhostPos)==2:
    #             score = score+5
    #         if l1Dist(newPos,newGhostPos)==1:
    #             score = score+10
    #         if l1Dist(newPos,newGhostPos)==0:
    #             score = score+100
    #
    # # incentivize moving
    # if curPos == newPos:
    #     score = score-1
    #
    # return score




# Abbreviation
better = betterEvaluationFunction
