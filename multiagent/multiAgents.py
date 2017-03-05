# multiAgents.py
# --------------
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"

        ## Used simple logic of checking distance from ghost and distance from food

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        foodDistance = []
        ghostDistance = []
        minGhostDistance = 0
        minFoodDistance = 0

        for food in newFood:
            foodDistance.append(manhattanDistance(newPos, food))

        if (len(foodDistance) > 0):
            minFoodDistance = min(foodDistance)

        if (len(newGhostStates) > 0):
            for ghost in newGhostStates:
                ghostDistance.append(manhattanDistance(newPos, ghost.getPosition()))
            minGhostDistance = min(ghostDistance)

            if (minGhostDistance == 0):
                return -99999

        totalcost = successorGameState.getScore() + minGhostDistance - minFoodDistance
        return totalcost

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
        """
        "*** YOUR CODE HERE ***"

        ## Implemented the algorithm minimax with the help of helper function minimax
        ## to call maximizer and minimizer passed true and false respectively
        ## in minimax function

        maxScore = float('-inf')
        nextAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            prevScore = maxScore
            maxScore = max(maxScore,
                           self.minimax(gameState, gameState.generateSuccessor(0, action), self.depth, 0, False))
            if maxScore > prevScore:
                nextAction = action
        return nextAction

    def minimax(self, gameState, state, depth, numGhost, isMaxmizer):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if isMaxmizer:
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, self.minimax(gameState, state.generateSuccessor(0, action), depth, 0, False))
            return v

        else:
            v = float('inf')
            numGhost += 1
            for action in state.getLegalActions(numGhost):
                if numGhost < (gameState.getNumAgents() - 1):
                    v = min(v,
                            self.minimax(gameState, state.generateSuccessor(numGhost, action), depth, numGhost, False))
                else:
                    v = min(v, self.minimax(gameState, state.generateSuccessor(numGhost, action), (depth - 1), numGhost,
                                            True))
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

        ## Implemented the algorithm alphabeta pruning with the help of helper function pruning
        ## to call maximizer and minimizer passed true and false respectively
        ## in pruning function

        alpha = float('-inf')
        beta = float('inf')
        maxScore = float('-inf')
        nextAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            prevScore = maxScore
            maxScore = max(maxScore,
                           self.pruning(gameState, gameState.generateSuccessor(0, action), alpha, beta, self.depth, 0,
                                        False))
            if maxScore > prevScore:
                nextAction = action
            if maxScore > beta:
                return nextAction
            alpha = max(alpha, maxScore)
        return nextAction

    def pruning(self, gameState, state, alpha, beta, depth, numGhost, isMaxmizer):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if isMaxmizer:
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, self.pruning(gameState, state.generateSuccessor(0, action), alpha, beta, depth, 0, False))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        else:
            v = float('inf')
            numGhost += 1
            for action in state.getLegalActions(numGhost):
                if numGhost < (gameState.getNumAgents() - 1):
                    v = min(v, self.pruning(gameState, state.generateSuccessor(numGhost, action), alpha, beta, depth,
                                            numGhost, False))
                else:
                    v = min(v,
                            self.pruning(gameState, state.generateSuccessor(numGhost, action), alpha, beta, (depth - 1),
                                         numGhost, True))

                if v < alpha:
                    return v
                beta = min(beta, v)
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
        ## Implemented the algorithm exectimax with the help of helper function expectimax
        ## to call maximizer and minimizer passed true and false respectively
        ## in expectimax function

        maxScore = float('-inf')
        nextAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            prevScore = maxScore
            maxScore = max(maxScore,
                           self.expectimax(gameState, gameState.generateSuccessor(0, action), self.depth, 0, False))
            if maxScore > prevScore:
                nextAction = action
        return nextAction

    def expectimax(self, gameState, state, depth, numGhost, isMaxmizer):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if isMaxmizer:
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, self.expectimax(gameState, state.generateSuccessor(0, action), depth, 0, False))
            return v

        else:
            v = 0.0
            numGhost += 1
            for action in state.getLegalActions(numGhost):
                if numGhost < (gameState.getNumAgents() - 1):
                    v = v + self.expectimax(gameState, state.generateSuccessor(numGhost, action), depth, numGhost,
                                            False)
                else:
                    v = v + self.expectimax(gameState, state.generateSuccessor(numGhost, action), (depth - 1), numGhost,
                                            True)
            return v / len(state.getLegalActions(numGhost))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

  DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    """
      ################################################################################
      ################################################################################
            For evaluation purpose focused on four major pints:
      1. Maximum manhattan distance of existing food from pacman
         it tells about how pacman is far from reaching his goal
      2. Minimum manhattan distance of ghost which is not scared from pacman
         As we need to make sure pacman should not die
      3. Minimum manhattan distance of ghost which is scared from pacman
         As we can eat that pacman without any problem
      4. Finally multiplied with factor according to importance
      ################################################################################
      ################################################################################
    """

    if currentGameState.isWin():
        return 99999

    if currentGameState.isLose():
        return -99999

    currentLoc = currentGameState.getPacmanPosition()
    ghostLoc = currentGameState.getGhostStates()
    foodLoc = currentGameState.getFood().asList()

    # calculating maximum distance of food from current pacman location
    maxdistance = MaximumManhattanDistance(currentLoc,foodLoc)
    minGhostDist = MinimumManhattanDistance(currentLoc, filter(lambda x: x.scaredTimer == 0, ghostLoc))
    minScaredGhostDist = MinimumManhattanDistance(currentLoc, filter(lambda x: x.scaredTimer > 0, ghostLoc))

    totalScore = scoreEvaluationFunction(currentGameState) + (-2 * maxdistance) + (-2.5 * ( minGhostDist)) + (2 * minScaredGhostDist)

    return totalScore

def MaximumManhattanDistance(currentState, foodList):
    distanceList = [0]
    for nextfood in foodList:
        distanceList.append(util.manhattanDistance(currentState, nextfood))
    return max(distanceList)

def MinimumManhattanDistance(currentState, ghostStates):
    if len(ghostStates) == 0:
        return 0
    else:
        distanceList = []
        for ghost in ghostStates:
            distanceList.append(util.manhattanDistance(currentState, ghost.getPosition()))
        return min(distanceList)

# Abbreviation
better = betterEvaluationFunction

