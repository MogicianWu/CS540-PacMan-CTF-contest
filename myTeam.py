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
from game import Directions, Agent, Actions
import game
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
    CaptureAgent.__init__(self, index, timeForComputing = .1)
    self.numParticles = 1000
    
  
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
      self.carry = 0
      self.invaders = Queue()
      self.enemyIndexs = self.getOpponents(gameState)
      self.myIndexes = self.getTeam(gameState)
      self.enStart = gameState.getInitialAgentPosition(self.enemyIndexs[0])
      self.bias = 0
      self.percentage = len(self.observationHistory)/(float)(300)
      if self.start[0] > self.enStart[0]:
          self.bias = 1
      else:
          self.bias = -1 
      self.middleX = (int) ( (self.start[0]+self.enStart[0])/2 + self.bias)
      self.gridWidth = (int) (self.start[1]+self.enStart[1])
      self.middleY = (int)(self.gridWidth*random.random())
      while gameState.hasWall(self.middleX,self.middleY):
          self.middleY = (int)(self.gridWidth*random.random())
      self.oldTarget = (self.middleX,self.middleY)
      self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
      self.particleList = []
      for enIndex in self.enemyIndexs:
          self.particleList.append({})
          if enIndex < 2:
              enIndex = 0
          else:
              enIndex = 1
          self.initializeUniformly(gameState,enIndex)

  def observe(self,gameState):
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
      
      #get parameters base on current state
      self.currScore = self.getScore(gameState)
      self.currPos = gameState.getAgentState(self.index).getPosition()
      self.twoFoods = self.twoFs(gameState)
      self.offenseFs = self.getFood(gameState).asList() 
      self.defenceFs = self.getFoodYouAreDefending(gameState).asList()
      for index in self.myIndexes:
          if index != self.index:
              self.DToTeam = self.getMazeDistance(self.currPos,gameState.getAgentState(index).getPosition())       
      self.closestG = None
      self.seenEn = [gameState.getAgentState(i) for i in self.enemyIndexs]
      
      #udpate enemy locations if seen
      for i in self.enemyIndexs:
          pos = gameState.getAgentState(i).getPosition()
          if pos!= None:
              if i < 2:
                  enIndex = 0
              else:
                  enIndex = 1
              self.updateExactLoc(pos,enIndex)
      
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
    
      #get parameters for inference     
      for enemyIndex in self.enemyIndexs:
          if enemyIndex <2:
              enIndex = 0
          else:
              enIndex = 1
          self.observeNoise(gameState.getAgentDistances()[enemyIndex], gameState,enIndex)
     
      dists = []
      for enemyIndex in self.enemyIndexs:
          d = util.Counter()
          if i == enemyIndex:
              if enemyIndex < 2:
                  enIndex = 0
              else:
                  enIndex = 1
              d =  self.getBeliefDistribution(enIndex)
              dists.append(d)        
      self.displayDistributionsOverPositions(dists)
      for enemyIndex in self.enemyIndexs:
          if enemyIndex <2:
              enIndex = 0
          else:
              enIndex = 1
          self.elapseTime(gameState,enIndex) 
      
      self.observeD = self.calcObserveD()
      
  def chooseAction(self, gameState):     
      #initialize last state and some parameters
      self.legalActions = gameState.getLegalActions(self.index)
      resultAction = None   
      self.observe(gameState)
      # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)  		      
      
      #update the nextStates after picking the bestActions;         
      resultAction = random.choice(self.ultronBook(gameState))
      nextState =  self.getSuccessor(gameState, resultAction)
      nextPos = nextState.getAgentState(self.index).getPosition()
      
      #make sure not bumping into ghosts when getting to the middle
      if self.DToGhosts < 4:
          if self.getMazeDistance(nextPos,self.closestG) < self.getMazeDistance(self.currPos,self.closestG):
              resultAction = random.choice(self.avoidTarget(gameState,self.legalActions,self.closestG))
              nextState =  self.getSuccessor(gameState, resultAction)
              nextPos = nextState.getAgentState(self.index).getPosition()
           
      #calculate next state's foods base on action selected      
      nextOffenseFs = self.getFood(nextState).asList() 
      nextDefenceFs = self.getFoodYouAreDefending(nextState).asList()          
      if len(self.offenseFs) >  len(nextOffenseFs):
          self.carry += (len(self.offenseFs) - len(nextOffenseFs))
      elif len(nextOffenseFs) > len(self.offenseFs) or self.getScore(gameState) < self.getScore(nextState):
          self.carry = 0     
          
      gotCha = len(nextDefenceFs) > len(self.defenceFs)
      if gotCha:
		  enIndex = 0
		  for i in self.enemyIndexs:
			   if gameState.getAgentState(i).getPosition() == nextPos:
				   if i < 2:
					   enIndex = 0
				   else:
					   enIndex = 1
				   self.updateExactLoc(nextPos,enIndex) 
	
      return resultAction
  
  def ultronBook(self,gameState):
      #mannual training;
      target = None
      middle = True       
      #offensive agent; get the closest Food if there isn't ghosts near you and you have enough carry on you; otherwise run back
      if self.observeD > 30:
          middle = False
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
              middle = False
                           
      #defensive agent; go to last known closest enmey location if they are close and we are not scared
      elif self.DToSeenInvaders < 6 and self.currState.getAgentState(self.index).scaredTimer == 0:   
          target = self.ClosestTarget(self.currPos,self.seenInvaders)
          middle = False  
      elif self.DToInvaders < 10  and self.currState.getAgentState(self.index).scaredTimer == 0:  
          target = self.ClosestTargetQ(self.currPos,self.invaders)
          middle = False    
          
      #if not on offense or defense, patrol in the middle; if everything is eaten run back as well  
      if middle or target == None or len(self.offenseFs) <= 2:   
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
 
  def getPositionDistribution(self, enemyPos, gameState):
      """
      Assume equal likelihood for all possible actions of our enemy
      """
      dist = util.Counter()
      neighbors = game.Actions.getLegalNeighbors(enemyPos,gameState.getWalls())
      for n in neighbors:
          dist[n] = 1
      dist.normalize()
      return dist
  
  def initializeUniformly(self, gameState,enIndex):
      particles = []
      for i in range(self.numParticles):
          particles.append(self.legalPositions[i%len(self.legalPositions)])   
      self.particleList[enIndex] = particles
   
  def observeNoise(self,noisyDistance, gameState,enIndex):
      oldWeights = self.getBeliefDistribution(enIndex)
      newWeights = util.Counter()
      for i in range(self.numParticles):
          particle = util.sample(oldWeights)
          trueDistance = self.getMazeDistance(particle,self.currPos)
          if  gameState.getDistanceProb(trueDistance,noisyDistance) > 0:
              newWeights[particle] = gameState.getDistanceProb(trueDistance,noisyDistance)*oldWeights[particle]
      if newWeights.totalCount()==0:
          self.initializeUniformly(gameState,enIndex)
      else:
          newWeights.normalize()
          for i in range(self.numParticles):
              self.particleList[enIndex][i] = util.sample(newWeights)
            
  def elapseTime(self,gameState,enIndex):
      newWeights = util.Counter()
      belief = self.getBeliefDistribution(enIndex)
      for oldPos in self.legalPositions:
          newPosDist = self.getPositionDistribution(oldPos,gameState)
          for newPos, prob in newPosDist.items():
              newWeights[newPos]+= prob*belief[oldPos]
      if newWeights.totalCount()==0:
          self.initializeUniformly(gameState,enIndex)        
      else:        
          distribution = []
          for pos in self.legalPositions:
              distribution.append(newWeights[pos])
          self.particleList[enIndex] = util.nSample(distribution,self.legalPositions,self.numParticles)
  
  def updateExactLoc(self,pos,enIndex):
      particles = []
      for i in range(self.numParticles):
		  particles.append(pos)
      self.particleList[enIndex] = particles
  
  def updateApproximate(self,pos):
	  copy = self.particleList[:]
	  for i in range(0,2):
		  for particle in copy[i]:
			  if particle[0] == pos[0]:
				  index = (int)(random.uniform(0,self.numParticles))
				  self.particleList[i][index] = pos

				 
  def getBeliefDistribution(self,enIndex):
      belief = util.Counter()
      for particle in self.particleList[enIndex]:
          belief[particle] += 1
      belief.normalize()
      return belief
  
  def calcObserveD(self):
	  distance = 0
	  observeD = float('inf')    
	  for i in range(2):
		  for pos in self.getBeliefDistribution(i).keys():
			  distance = distance + self.getMazeDistance(self.currPos,pos)*   self.getBeliefDistribution(i)[pos]
		  distance = min(observeD,distance)
	  return distance
