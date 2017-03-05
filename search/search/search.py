# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    ## Will be using stack as DFS demands
    ##  will be pushing directions only along each path and retrieve those direction when goalstate achieved
    fringe = util.Stack()
    fringe.push((problem.getStartState(), []))
    visitedSet = []
    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode[0]):
            return currentNode[1]


        if currentNode[0] not in visitedSet:
            visitedSet.append(currentNode[0])
        else:
            continue

        for successors in problem.getSuccessors(currentNode[0]):
            if successors[0] not in visitedSet:
                 newNode =(successors[0], currentNode[1] + [successors[1]])
                 fringe.push(newNode)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    ## Will be using queue as BFS demands
    ##  will be pushing directions only along each path and retrieve those direction when goalstate achieved
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))
    visitedSet=[]
    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode[0]):
            return currentNode[1]

        if currentNode[0] not in visitedSet:
            visitedSet.append(currentNode[0])
        else:
            continue

        for successors in problem.getSuccessors(currentNode[0]):
            if successors[0] not in visitedSet:
                newNode = (successors[0], currentNode[1] + [successors[1]])
                fringe.push(newNode)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    ## Will be using priorityqueue as we now need to maintain queue along with cost
    ##  will be pushing directions and cost along each path and retrieve those direction when goalstate achieved
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(),[],0),0)
    visitedSet = []
    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode[0]):
            return currentNode[1]

        if currentNode[0] not in visitedSet:
            visitedSet.append(currentNode[0])
        else:
            continue

        for successors in problem.getSuccessors(currentNode[0]):
            if successors[0] not in visitedSet:
                costPath =currentNode[-1] + successors[-1]
                newNode = (successors[0], currentNode[1] + [successors[1]],costPath)
                fringe.push(newNode,costPath)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    ## Will be using priorityqueue as we now need to maintain queue along with cost provided by caculated cost and heuristic cost
    ##  will be pushing directions and total cost along each path and retrieve those direction when goalstate achieved
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), []), heuristic(problem.getStartState(),problem))
    visitedSet = []
    while not fringe.isEmpty():
        currentNode = fringe.pop()


        if problem.isGoalState(currentNode[0]):
            return currentNode[1]



        if currentNode[0] not in visitedSet:
            visitedSet.append(currentNode[0])
        else:
            continue


        for successors in problem.getSuccessors(currentNode[0]):
            if successors[0] not in visitedSet:
                costHeuristic=heuristic(successors[0],problem)
                costPath = problem.getCostOfActions(currentNode[1] + [successors[1]])
                totalCost= costHeuristic + costPath
                newNode = (successors[0], currentNode[1] + [successors[1]])
                fringe.push(newNode, totalCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
