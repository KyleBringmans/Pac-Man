# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):

    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """
    # TODO add class-variable to be assigned to the matrix of shortest paths
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        ghostStates = state.getGhostStates()
        sTime = state.getScaredTime()
        n = 3  # distance instead of 1
        features = util.Counter()


        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features["scared"] = (sTime - (self.avgScaredTime(ghostStates))) / (sTime*1.0)
        features["bias"] = 1.0
        #inputList = zip(ghosts,[(x,y)]*len(ghosts))
        #ghostDistances = map(lambda q: self.shortestPath(q[0]),inputList)

        # --------------------------------------------------------------------------------------------------------------

        #features["#-of-ghosts-n-steps-away"] = len(filter(lambda t: t < n, map(lambda q: self.shortestPath(q[0][0],q[0][1],q[1][0],q[1][1],walls),inputList)))
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        notScared = list(filter(lambda q: q[1].scaredTimer == 0,zip(ghosts, ghostStates)))
        features["#-of-not-scared-ghosts-n-steps-away"] = sum(self.euclDist(x, y, g[0][0], g[0][1]) < n for g in notScared)
        #features["#-of-ghosts-n-steps-away"] = sum((next_x,next_y) in Actions.getLegalNeighbors(ns[0],walls) for ns in notScared)

        # --------------------------------------------------------------------------------------------------------------

        features["#-of-ghosts-scared"] = len(filter(lambda q: self.euclDist(x,y,q[0],q[1]) < n,ghosts)) - len(notScared)

        # --------------------------------------------------------------------------------------------------------------

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0


        capsules = map(lambda q: [False]*len(q),food)
        for cap in state.getCapsules():
            capsules[cap[0]][cap[1]] = True

        distFood = closestFood((next_x, next_y), food, walls)
        distCapsule = closestFood((next_x, next_y), capsules, walls)
        if distFood is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(distFood) / (walls.width * walls.height)
        if distCapsule is not None:
            features["closest-capsule"] = float(distCapsule) / (walls.width * walls.height)



        surface = walls.height*walls.width*1.0
        features["hallway-0"] = (self.inHallwayRec(y+1, x,   (y,x), walls) if self.notWall(y+1, x,   walls) else 0)/surface
        features["hallway-1"] = (self.inHallwayRec(y,   x+1, (y,x), walls) if self.notWall(y,   x+1, walls) else 0)/surface
        features["hallway-2"] = (self.inHallwayRec(y-1, x,   (y,x), walls) if self.notWall(y-1, x,   walls) else 0)/surface
        features["hallway-3"] = (self.inHallwayRec(y,   x-1, (y,x), walls) if self.notWall(y,   x-1, walls) else 0)/surface




        features.divideAll(10.0)


        return features

    # ------------------------------------------------------------------------------------------------------------------

    def avgScaredTime(self,states):
        tot = 0
        for i in range(len(states)):
           tot += states[i].scaredTimer
        a = tot/len(states)
        return a

    # ------------------------------------------------------------------------------------------------------------------

    def inHallway(self, x, y, origin, walls):
        nbrs = self.getNeighboursSimple(x, y, walls)
        # don't count doubles (spots counted in previous iteration
        nbrs = filter(lambda q: q != origin, nbrs)
        # check if there are the correct # of nbr walls -> hallway
        if len(nbrs) == 2:
            fst = nbrs[0]
            snd = nbrs[1]
            a = 1 + self.inHallwayRec(fst[0], fst[1], (x, y), walls)
            b = self.inHallwayRec(snd[0], snd[1], (x, y), walls)
            return a + b
        elif len(nbrs) == 1:
            fst = nbrs[0]
            return 1 + self.inHallwayRec(fst[0], fst[1], (x, y), walls)
        elif len(nbrs) == 0:
            return 1
        else:
            return 0

    def inHallwayRec(self,x,y,origin,walls):
        nbrs = self.getNeighboursSimple(x,y,walls)
        # don't count doubles (spots counted in previous iteration)
        nbrs = filter(lambda q: q != origin, nbrs)
        # check if there are the correct # of nbr walls -> hallway
        if len(nbrs) == 1:
            fst = nbrs[0]
            return 1 + self.inHallwayRec(fst[0], fst[1], (x,y), walls)
        elif len(nbrs) == 0:
            return 1
        else:
            return 0

    def getDirectionalNeighbour(self,x,y,direction):
        return x + direction[0], y + direction[1]

    def euclDist(self,x1,y1,x2,y2):
        return math.sqrt(((x1-x2)**2) + ((y1-y2)**2))

    # ------------------------------------------------------------------------------------------------------------------

    def shortestPath(self,start_x,start_y,dest_x,dest_y,walls): # A* algorithm
        if not self.notWall(start_x,start_y,walls) or not self.notWall(dest_x,dest_y,walls):
            return []
        visited = set()
        q = util.PriorityQueue()
        q.push([(start_x,start_y),[],0],0)
        while not q.isEmpty():
            loc,path,cost = q.pop()
            if (dest_x,dest_y) == loc:
                return path
            if loc not in visited:
                visited.add(loc)
                for (x,y) in self.getNeighboursSimple(loc[0],loc[1],walls):
                    if (x,y) not in visited:
                        backwardCost = 1 + cost
                        forwardCost = self.euclDist(x,y,dest_x,dest_y)
                        fx = backwardCost + forwardCost
                        q.push([(x,y),path + [(x,y)],backwardCost],fx)

    def notWall(self,x,y,walls):
        y = int(y)
        w = walls[y]
        x = int(x)
        return not w[len(w) - 1 - x]



    def getNeighboursSimple(self,x,y,walls):
        width = walls.width
        height = walls.height
        nbrs = self.generateAllNeighboursSimple(x,y)
        nbrs = filter(lambda q: q[1] < width >= 0 and q[0] < height >= 0, nbrs) # keep nbrs in grid
        nbrs = filter(lambda q: self.notWall(q[0],q[1],walls),nbrs) # remove neighbours that aren't walls
        return nbrs

    def getAllNeighboursSimple(self,x,y,walls):
        width = walls.width
        height = walls.height
        nbrs = self.generateAllNeighboursSimple(x, y)
        nbrs = filter(lambda q: q[1] < width >= 0 and q[0] < height >= 0, nbrs)  # keep nbrs in grid
        return nbrs

    def generateAllNeighboursSimple(self,x,y):
        return [(x+1,y),(x,y+1),(x-1,y),(x,y-1)]

    # ------------------------------------------------------------------------------------------------------------------

    def calculateCorners(self,path,walls):
        corners = 0
        if len(path) < 3:
            return 0
        for i in range(0,len(path)-2):
            bCorner = path[i]
            corner = path[i+1]
            aCorner = path[i+2]
            if self.euclDist(bCorner[0],bCorner[1],aCorner[0],aCorner[1]) == math.sqrt(2):
                if self.isCorner([bCorner,corner,aCorner],walls):
                    corners += 1
        return corners

        # TODO ehhhh is this correct?

    def isCorner(self, corner, walls):
        if self.wallNeighbours(corner[0], walls) > 0 and self.wallNeighbours(corner[2], walls) > 0:
            return True
        else:
            return False

    def wallNeighbours(self,pos,walls):
        nbrs = self.getAllNeighboursSimple(pos[0],pos[1],walls)
        w = 0
        for p in nbrs:
            if not self.notWall(p[0],p[1],walls): # so if it is a wall
                w += 1
        return w

