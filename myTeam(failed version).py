# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions, Agent
import game
import pickle
from util import Queue

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,first = 'UltronPac', second = 'UltronPac',**args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

#Below are the Q learning agents
class UltronPac(CaptureAgent):
  """
  One agent to serve as both the defender and the attacker in the game
  Idea: Use Q learning to play against itself before the CTF tournament 
  """
  def __init__(self, index, timeForComputing = .1 ):
    self.weights = util.Counter()
    self.loadWeights()
    CaptureAgent.__init__(self, index, timeForComputing = .1)
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.numTraining = 0
    self.epsilon = 0.1
    self.alpha = 0.2
    self.discount = 0.8
    
  def getFeatures(self, gameState, action): 
      features = util.Counter() 
      features["bias"] = 1.0
      
      #features base on past state
      features["last-score"] = self.lastScore
      features["last-difference"] = self.lastDiff
      
      #features base on current state
      features["num-of-capsule"] = len(self.getCapsules(gameState))
      features["scaredTimer"] = self.currState.getAgentState(self.index).scaredTimer
      features["percentage-of-game-played"] = self.currPercentage
      features["curr-score"] = self.currScore
      
      #calculate the distance to the closest food and farthest food
      if len(self.twoFs(gameState))==1:
          dist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),self.twoFs(gameState)[0])
          features["closest-food"] = dist
      elif len(self.twoFs(gameState))==2:
          dist1 = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),self.twoFs(gameState)[0])
          features["closest-food"] = dist1
          dist2 = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),self.twoFs(gameState)[1])
          features["farthest-food"] = dist2
      
      # use number of foods left as a feature
      features["num-of-food-attack"] = len(self.offenseFs)
      features["pre-num-of-food-attack"] = len(self.lastOffenseFs)
      features["num-of-food-to-defend"] =  len(self.defenceFs)
      features["pre-num-of-food-to-defend"] = len(self.lastDefenceFs)
      if len(self.getCapsules(gameState))==1:
          features["distance-to-capsule"] = self.disToCapsules
      if self.DToInvaders < 100:
          features["distance-to-invaders"] = self.DToInvaders 
      if self.DToSeenInvaders < 100:
          features["distance-to-seen-invaders"] = self.DToSeenInvaders
      if self.DToGhosts < 100:
          features["distance-to-ghosts"] = self.DToGhosts
      if self.DToScared < 100:
          features["distance-to-sacred-ghosts"] = self.DToScared 
    
      features["noisy-distance-to-enemy1"]= self.noisyDisToEn1 
      features["noisy-distance-to-enemy1"]= self.noisyDisToEn1 
      features["distance-between-teammates"] =  self.DToTeam
      
      #features base on nextState
      nextState =  self.getSuccessor(gameState, action)
      nextPos = nextState.getAgentState(self.index).getPosition()
      nextOffenseFs = self.getFood(nextState).asList() 
      nextDefenceFs = self.getFoodYouAreDefending(nextState).asList()
      gotCha = len(nextDefenceFs) > len (self.defenceFs)  
      
      for index in self.myIndexes:
          if index != self.index:
              nextDToTeam = self.getMazeDistance(nextPos,gameState.getAgentState(index).getPosition())       
              
      features["next-noisy-distance-to-enemy1"]= nextState.getAgentDistances()[self.enemyIndexes[0]]
      features["next-noisy-distance-to-enemy2"]= nextState.getAgentDistances()[self.enemyIndexes[1]]
      features["next-distance-between-teammates"] =  nextDToTeam
      features["next-score"] = self.getScore(nextState)
      features["next-offense-food"] = len(nextOffenseFs)
      features["next-defence-food"] = len(nextDefenceFs)
      features["next-sacredTimer"] = nextState.getAgentState(self.index).scaredTimer
      if len(self.getCapsules(nextState))==1:
         features["next-d-to-capsule"] = self.getMazeDistance(nextPos,self.getCapsules(nextState)[0])
      if len(self.seenInvaders)!=0:
          features["next-d-to-seen-invaders"] = self.ClosestD(nextPos,self.seenInvaders)
      if self.size(self.invaders)!=0:
          features["next-d-to-invaders"]= self.ClosestDQ(nextPos,self.invaders)
      if len(self.seenGhosts)!=0:
          features["next-d-to-seen-ghosts"] = self.ClosestD(nextPos,self.seenGhosts)
      if len(self.seenScared)!=0:
          features["next-d-to-closest-sacred"] = self.ClosestD(nextPos,self.seenScared)       
      if len(self.observationHistory)> 10 and self.currPos == self.start:
          features["they-got-me"] = 1
      if gotCha:
		  features["I-got-them"] = 1
      if len(self.offenseFs) >  len(nextOffenseFs):
          features["add-carry"] = len(self.offenseFs) - len(nextOffenseFs)
      elif len(nextOffenseFs) > len(self.offenseFs) or self.getScore(gameState) < self.getScore(nextState):
          features["minus-carry"] = 1        
      
      return features
      
  def twoFs(self,gameState):
      foods = self.getFood(gameState).asList()
      if len(foods)<=1:
          if len(foods)==1:
              self.MostCloseDToF = self.getMazeDistance(foods[0], gameState.getAgentPosition(self.index))
              self.MostFarDToF = self.MostCloseDToF
          return foods
          
      Dis = [self.getMazeDistance((self.currPos),food) for food in foods]
      twoFs = []
      minimum = min(Dis)
      maximum = max(Dis)
      close = [f for f, v in zip(foods,Dis) if v == minimum]
      far = [f for f, v in zip(foods,Dis) if v == maximum]
      twoFs.append(random.choice(close))
      twoFs.append(random.choice(far))
      return twoFs

  
  def registerInitialState(self, gameState,**args):
      
      """
      This method handles the initial setup of the
      agent to populate useful fields (such as what team
      we're on).

      A distanceCalculator instance caches the maze distances
      between each pair of positions, so your agents can use:
      self.distancer.getDistance(p1, p2)

      IMPORTANT: This method may run for at most 15 seconds.
      """

      '''
      Make sure you do not delete the following line. If you would like to
      use Manhattan distances instead of maze distances in order to save
      on initialization time, please take a look at
      CaptureAgent.registerInitialState in captureAgents.py.
      '''
      CaptureAgent.registerInitialState(self, gameState)

      '''
      Your initialization code goes here, if you need any.
      '''
      self.start = gameState.getAgentPosition(self.index)
      actionFn = gameState.getLegalActions(self.index)
      self.actionFn = actionFn     
      self.startEpisode()
      self.carry = 0
      self.invaders = Queue()
      self.enemyIndexes = self.getOpponents(gameState)
      self.myIndexes = self.getTeam(gameState)
      self.enStart = gameState.getInitialAgentPosition(self.enemyIndexes[0])
      self.bias = 0
      self.gotCha = False
      self.attack = False
      self.defend = False
      self.middle = False
      self.percentage = len(self.observationHistory)/(float)(300)
      if self.start[0] > self.enStart[0]:
          self.bias = 1
      else:
          self.bias = -1 
      self.middleX = (int) ( (self.start[0]+self.enStart[0])/2 + self.bias )
      self.gridWidth = (int) (self.start[1]+self.enStart[1])
      self.middleY = (int)(self.gridWidth*random.random())
      while gameState.hasWall(self.middleX,self.middleY):
          self.middleY = (int)(self.gridWidth*random.random())
      self.oldTarget = (self.middleX,self.middleY)
      if self.episodesSoFar == 0:
          print 'Beginning %d episodes of Training' % (self.numTraining)
    
  def chooseAction(self, gameState):     
      #initialize last state and some parameters
      self.legalActions = gameState.getLegalActions(self.index)
      resultAction = None   
      self.currState = gameState
      self.lastState = self.getPreviousObservation()
      if self.lastState == None:
		  self.lastState = gameState
      self.currPercentage = len(self.observationHistory)/(float)(300)
      self.DToSeenInvaders = float('inf')
      self.DToInvaders = float('inf')
      self.DToGhosts = float('inf')
      self.DToScared = float('inf')
      
      
      #parameters base on last state
      self.lastScore = self.getScore(self.lastState)
      self.lastOffenseFs = self.getFood(self.lastState).asList()
      self.lastDefenceFs = self.getFoodYouAreDefending(self.lastState).asList()
      self.lastPos = self.lastState.getAgentState(self.index).getPosition()
      for index in self.myIndexes:
          if index != self.index:
              self.lastDToTeam = self.getMazeDistance(self.lastPos,gameState.getAgentState(index).getPosition())   
      self.lastDiff = len(self.lastDefenceFs)-len(self.lastOffenseFs)
      
      #get parameter base on current state
      self.currScore = self.getScore(gameState)
      self.currPos = gameState.getAgentState(self.index).getPosition()
      self.twoFoods = self.twoFs(gameState)
      self.offenseFs = self.getFood(gameState).asList() 
      self.defenceFs = self.getFoodYouAreDefending(gameState).asList()
      self.currDiff = len(self.defenceFs)-len(self.offenseFs)
      for index in self.myIndexes:
          if index != self.index:
              self.DToTeam = self.getMazeDistance(self.currPos,gameState.getAgentState(index).getPosition())       
      self.closestG = None
      self.noisyDisToEn1 = gameState.getAgentDistances()[self.enemyIndexes[0]]
      self.noisyDisToEn2 = gameState.getAgentDistances()[self.enemyIndexes[1]]
      self.noisyClosestD = min(self.noisyDisToEn1,self.noisyDisToEn2) 
      self.seenEn = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      self.seenGhosts = [a.getPosition() for a in self.seenEn if a.getPosition() != None and not a.isPacman and a.scaredTimer==0]
      self.seenInvaders = [a.getPosition() for a in self.seenEn if a.getPosition() != None and a.isPacman]
      self.seenScared = [a.getPosition() for a in self.seenEn if a.getPosition() != None and not a.isPacman and a.scaredTimer!=0]
      if len(self.getCapsules(gameState))==1:
         self.disToCapsules = self.getMazeDistance(self.currPos,self.getCapsules(gameState)[0])
      if len(self.seenInvaders)!=0:
          for i in self.seenInvaders:
              self.push(self.invaders,i)  
          self.DToSeenInvaders =  self.ClosestD(self.currPos,self.seenInvaders)
      if self.size(self.invaders)!=0:
          self.DToInvaders = self.ClosestDQ(self.currPos,self.invaders)
      if len(self.seenGhosts)!=0:
          self.DToGhosts = self.ClosestD(self.currPos,self.seenGhosts)
          self.closestG = self.ClosestTarget(self.currPos,self.seenGhosts)
      if len(self.seenScared)!=0:
          self.DToScared = self.ClosestD(self.currPos,self.seenScared)   
        
      if len(self.lastDefenceFs)-len(self.defenceFs) > 0:
		  self.updateInvaders(gameState) 
               
      # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
      if len(self.legalActions)==0:
          return resultAction          
        
      #pick action based on q values; if there are two values that are too close to each other or same use mannual training
      values = [self.evaluate(gameState, a) for a in self.legalActions]  
      print 'qvalues: ',values 
      maxValue = max(values)
      bestActions = [a for a, v in zip(self.legalActions, values) if v == maxValue] 
      print bestActions
      if util.flipCoin(self.epsilon):
          bestActions = self.legalActions
      
      fuck = False
      smallestDiff = float('inf')
      for value in values:
          if maxValue - value < smallestDiff and maxValue != value:
              smallestDiff = maxValue - value
      if maxValue !=0:
          if (float)(smallestDiff) < 0.1 or len(bestActions)>=2:
              print 'small diff was: ',smallestDiff
              print 
              print 
              bestActions = self.trainingBook(gameState)
              fuck = True
    
    
      if sum(self.weights.values()) == 0:
          bestActions = self.trainingBook(gameState)  
      
      if self.getMazeDistance(self.start,self.currPos) < 16:
          bestActions = self.trainingBook(gameState)
      
      if not fuck:
          print 'yeah'
          print 'bestActions from q value are: ',bestActions
          print 
          print 
      
      #update the nextStates after picking the bestActions;         
      resultAction = random.choice(bestActions)
      nextState =  self.getSuccessor(gameState, resultAction)
      
      #make sure not bumping into ghosts when getting to the middle
      if self.DToGhosts < 4:
          nextPos = nextState.getAgentState(self.index).getPosition()
          if self.getMazeDistance(nextPos,self.closestG) < self.getMazeDistance(self.currPos,self.closestG):
              resultAction = random.choice(self.avoidTarget(gameState,self.legalActions,self.closestG))
              nextState =  self.getSuccessor(gameState, resultAction)
              
      nextOffenseFs = self.getFood(nextState).asList() 
      nextDefenceFs = self.getFoodYouAreDefending(nextState).asList()
      gotCha = len(nextDefenceFs) > len (self.defenceFs) 
      if len(self.seenInvaders)!=0:
          for i in self.seenInvaders:
              if i == nextState.getAgentState(self.index).getPosition():
                  gotCha = True             
    
      deltaReward =  self.currDiff - self.lastDiff + self.currScore - self.getScore(self.lastState)
      if len(self.observationHistory)> 10 and self.currPos == self.start:
          deltaReward -= 4
      if self.gotCha:
		  deltaReward += 4
		 
      if len(self.offenseFs) >  len(nextOffenseFs):
          self.carry += (len(self.offenseFs) - len(nextOffenseFs))
      elif len(nextOffenseFs) > len(self.offenseFs) or self.getScore(gameState) < self.getScore(nextState):
          self.carry = 0
      
      if self.lastAction!= None:
		  self.observeTransition(self.lastState,self.lastAction,gameState,deltaReward*10000000000000)
	  
      self.lastAction = resultAction
      return resultAction
    
  
  def trainingBook(self,gameState):
      print 'train book'
      #mannual training;
      target = None
      self.middle = True       
      #offensive agent; get the closest Food if there isn't ghosts near you and you have enough carry on you; otherwise run back
      if self.carry < 4:
          self.attack = True
          self.middle = False
          if len(self.twoFoods)== 1:
              target = self.twoFoods[0]
          elif len(self.twoFoods)==2:
              if self.DToTeam > 5:
                  target = self.twoFoods[0]
              else:
                  if self.index < 2: 
                      target = self.twoFoods[0]
                  else:
                      target = self.twoFoods[1]        
      
      #eat the ghost if it's sacred and close
      if len(self.seenScared)!=0:
          if self.ClosestD(self.currPos,self.seenScared) < 3:
              target = self.ClosestTarget(self.currPos,self.seenScared)
                           
      #defensive agent; go to last known closest enmey location if they are close and we are not scared
      elif self.DToSeenInvaders < 6 and self.currState.getAgentState(self.index).scaredTimer == 0:   
          target = self.ClosestTarget(self.currPos,self.seenInvaders)
          self.defend = True
          self.middle = False  
      elif self.DToInvaders < 10  and self.currState.getAgentState(self.index).scaredTimer == 0:  
          target = self.ClosestTargetQ(self.currPos,self.invaders)
          self.defend = True
          self.middle = False    
          
      #if not on offense or defense, patrol in the middle; if everything is eaten run back as well  
      if self.middle or target == None or len(self.offenseFs) <= 2:   
          #change target if we got to old target; don't change if we haven't gotten there yet
          if self.oldTarget == self.currPos:
              newY = (int)(self.gridWidth*random.random())
              while gameState.hasWall(self.middleX,newY):
                  newY = (int)(self.gridWidth*random.random())
              target = (self.middleX,newY)
          else:
              target = self.oldTarget    
          
      self.oldTarget = target   
      return self.getToTarget(gameState,self.legalActions,target)
    
  def getSuccessor(self, gameState, action):
      """
      Finds the next successor which is a grid position (location tuple).
      """
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != util.nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
      else:
          return successor

  def evaluate(self,gameState,action):
	  return self.getQValue(gameState,action)
           
  def update(self, state, action, nextState, reward):
      """
         Should update your weights based on transition
      """
      "*** YOUR CODE HERE ***"
      feature = self.getFeatures(state,action)
      weights = self.weights.copy()
      #update the weights of the feature; note in one update step we should be using the same old weights
      for key in feature.keys():
          if len(self.getLegalActions(nextState))!=0:
              weights[key] += self.alpha * (reward + self.discount * max( self.getQValue(nextState,a) for a in self.getLegalActions(nextState) ) - self.getQValue(state,action)) * feature[key]
          else:
              weights[key] += self.alpha * (reward - self.getQValue(state,action)) * feature[key]      
      weights.normalize()
		  
      self.weights = weights
      
  def getWeights(self):
      return self.weights

  def getQValue(self, state, action):
      """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
      """
      feature = self.getFeatures(state,action)
      return self.getWeights()*feature
  
  def final(self, state):
      """
        Called by Pacman game at the terminal state
      """
      self.observationFunction(state)
      self.stopEpisode()
      print 'episodes so far are: ',self.episodesSoFar
      # Make sure we have this var
      if not 'episodeStartTime' in self.__dict__:
          self.episodeStartTime = time.time()
      if not 'lastWindowAccumRewards' in self.__dict__:
          self.lastWindowAccumRewards = self.currScore
    
      NUM_EPS_UPDATE = 10
      if self.episodesSoFar % NUM_EPS_UPDATE == 0:
          print 'Reinforcement Learning Status:'
          windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
          if self.episodesSoFar <= self.numTraining:
              trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
              print '\tCompleted %d out of %d training episodes' % (
                     self.episodesSoFar,self.numTraining)
              print '\tAverage Rewards over all training: %.2f' % (
                      trainAvg)
          else:
              testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
              print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
              print '\tAverage Rewards over testing: %.2f' % testAvg
          print '\tAverage Rewards for last %d episodes: %.2f'  % (
                  NUM_EPS_UPDATE,windowAvg)
          print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
          self.lastWindowAccumRewards = 0.0
          self.episodeStartTime = time.time()

          
      if self.episodesSoFar == self.numTraining:
          msg = 'Training Done (turning off epsilon and alpha)'
          print '%s\n%s' % (msg,'-' * len(msg))
          
      
      CaptureAgent.final(self,state)

  
  def getLegalActions(self,state):
      """
        Get the actions available for a given
        state. This is what you should use to
        obtain legal actions for a state
      """
      return state.getLegalActions(self.index)

  def observeTransition(self,lastState,lastAction,currState,deltaReward):
      self.episodeRewards += deltaReward
      self.update(lastState,lastAction,currState,deltaReward)

  def startEpisode(self):
      """
        Called by environment when new episode is starting
      """
      self.lastState = None
      self.lastAction = None
      self.episodeRewards = 0.0

  def stopEpisode(self):
      """
        Called by environment when episode is done
      """
      if self.episodesSoFar < self.numTraining:
          self.accumTrainRewards += self.currScore
      else:
          self.accumTestRewards += self.currScore
      self.episodesSoFar += 1
      if self.episodesSoFar >= self.numTraining:
          # Take off the training wheels
          self.epsilon = 0.0    # no exploration
          self.alpha = 0.0      # no learning

  def isInTraining(self):
      return self.episodesSoFar < self.numTraining

  def isInTesting(self):
      return not self.isInTraining()
  
  def getToTarget(self,gameState,legalActions,target):
      disToDes = [self.getMazeDistance(self.getSuccessor(gameState,action).getAgentPosition(self.index),target) for action in legalActions]
      bestActions = [a for a, v in zip(legalActions,disToDes) if v == min(disToDes)]
      if len(bestActions)==0:
          return legalActions
      return bestActions
   
  def avoidTarget(self,gameState,legalActions,target):
      disToDes = [self.getMazeDistance(self.getSuccessor(gameState,action).getAgentPosition(self.index),target) for action in legalActions]
      bestActions = [a for a, v in zip(legalActions,disToDes) if v == max(disToDes)]
      if len(bestActions)==0:
          return legalActions
      return bestActions
      
  def size(self,queue):
      count = 0
      sameQ = Queue()
      while not queue.isEmpty():
          sameQ.push(queue.pop())
          count +=1
      while not sameQ.isEmpty():
          queue.push(sameQ.pop())    
    
      return count     
    
  def push(self,queue,item):
      while self.size(queue)>=2:
          queue.pop()
      queue.push(item)
  
  def peek(self,queue):
      item = None
      if self.size(queue)==1:
          item = queue.pop()
          queue.push(item)
      if self.size(queue)==2:
          item = queue.pop()
          item2 = queue.pop()
          queue.push(item)
          queue.push(item2)
      return item
 
  def peekLast(self,queue):
      item = None
      if self.size(queue)==1:
          item = queue.pop()
          queue.push(item)
      if self.size(queue)==2:
          item1 = queue.pop()
          item2 = queue.pop()
          queue.push(item1)
          queue.push(item2)
          item = item2
      
      return item
          
  def updateInvaders(self,gameState):
      for previous in self.lastDefenceFs:
          notEaten = False
          for curr in self.defenceFs:
              if previous==curr:
                  notEaten = True
          if not notEaten:
              self.push(self.invaders,previous) 
              
  def saveWeights(self,weights):
      dump_file = open("05-05.pkl", "wb")
      pickle.dump(weights, dump_file)
      dump_file.close()
      
  def loadWeights(self):
      in_file = open("05-05.pkl")
      file = pickle.load(in_file)
      self.weights = file
  
  
  def ClosestD(self,pos,targetList):
      Des = []
      for i in range(len(targetList)):
          Des.append(self.getMazeDistance(pos,targetList[i]))
      return min(Des)
  
  def ClosestDQ(self,pos,queue):
      Des = []
      sameQ = Queue()
      while not queue.isEmpty():
          item = queue.pop()
          Des.append(self.getMazeDistance(pos,item))
          sameQ.push(item)
      while not sameQ.isEmpty():
          queue.push(sameQ.pop())
      if len(Des) == 0:
          return float('inf')
      return min(Des)
  
  def ClosestTarget(self,pos,targetList):
      target = None
      minimum = float('inf')
      if len(targetList) == 0:
          return target
      for i in range(len(targetList)):
          if self.getMazeDistance(pos,targetList[i]) == self.ClosestD(pos,targetList):
              target = targetList[i]
      return target
  
  def ClosestTargetQ(self,pos,queue):
      target = None
      sameQ = Queue()
      while not queue.isEmpty():
         item = queue.pop()
         sameQ.push(item)
         if self.getMazeDistance(pos,item) == self.ClosestDQ(pos,queue):
             target = item
      return target
 
