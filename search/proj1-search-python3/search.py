# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # initialize the stack
    frontier = util.Stack()
    actions={}

    frontier.push(problem.getStartState());
    currentNode=(frontier.pop(),'',1);
    actions[currentNode] = []
    explored=[problem.getStartState()]

    while problem.isGoalState(currentNode[0]) == False:

        if problem.isGoalState(currentNode[0]):
            return actions[currentNode]

        for successor in problem.getSuccessors(currentNode[0]):
            if successor[0] not in explored:
                frontier.push(successor)
                if isinstance(actions[currentNode],list):
                    actions[successor] = actions[currentNode] + [successor[1]]
                else:
                    actions[successor] = [successor[1]]

        currentNode=frontier.pop();
        explored.append(currentNode[0])

    return actions[currentNode]


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    currentState = problem.getStartState();
    frontier = util.Queue();
    frontier_list = []
    actions={}

    actions[currentState] = []
    explored=[]

    while problem.isGoalState(currentState) == False:
        explored.append(currentState)
        if problem.isGoalState(currentState):
            return actions[currentState]
        for successor in problem.getSuccessors(currentState):
            if successor[0] not in explored and successor[0] not in frontier_list:
                frontier.push(successor[0])
                frontier_list.append(successor[0])
                actions[successor[0]] = actions[currentState] + [successor[1]]
        currentState=frontier.pop()
    return actions[currentState]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    frontier = util.PriorityQueue()
    explored=[]
    actions={}

    frontier.push(problem.getStartState(),0)
    node=(problem.getStartState(),'',0)
    actions[node[0]] = []
    i=True;
    while i==True:
        if frontier.isEmpty():
            print("Empty Frontier, Goal not Found.")
            return None
        node = frontier.pop()

        # for bizarre test behavior...
        if len(node) < 3:
            node = (node,'',0)

        if problem.isGoalState(node[0]):
            return actions[node[0]]
        explored.append(node[0])
        if node[0] not in actions.keys():
            print(node[0], 'no actions')
        for successor in problem.getSuccessors(node[0]):
            cost = problem.getCostOfActions(actions[node[0]]+[successor[1]])
            if (successor[0] not in explored) and (successor[0] not in [x[2][0] for x in frontier.heap]):
                frontier.push(successor, cost)
                actions[successor[0]] = actions[node[0]] + [successor[1]]
                explored.append(successor[0])
            elif (successor[0] in [x[2][0] for x in frontier.heap]) and (cost < problem.getCostOfActions(actions[successor[0]])):
                frontier.update(successor, cost)
                actions[successor[0]] = actions[node[0]] + [successor[1]]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    explored=[]
    actions={}

    frontier.push(problem.getStartState(),0)
    node=(problem.getStartState(),'',0)
    actions[node[0]] = []
    i=True;
    while i==True:
        if frontier.isEmpty():
            print("Empty Frontier, Goal not Found.")
            return None
        node = frontier.pop()

        # for bizarre test behavior...
        if len(node) < 3:
            node = (node,'',0)

        if problem.isGoalState(node[0]):
            return actions[node[0]]
        explored.append(node[0])
        if node[0] not in actions.keys():
            print(node[0], 'no actions')
        for successor in problem.getSuccessors(node[0]):
            cost = problem.getCostOfActions(actions[node[0]]+[successor[1]]) + heuristic(successor[0], problem)
            if (successor[0] not in explored) and (successor[0] not in [x[2][0] for x in frontier.heap]):
                frontier.push(successor, cost)
                actions[successor[0]] = actions[node[0]] + [successor[1]]
                explored.append(successor[0])
            elif (successor[0] in [x[2][0] for x in frontier.heap]) and (cost < problem.getCostOfActions(actions[successor[0]])+heuristic(successor[0], problem)):
                frontier.update(successor, cost)
                actions[successor[0]] = actions[node[0]] + [successor[1]]



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
