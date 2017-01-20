# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D() below.
#
# You are given a car in a grid with initial state
# init = [x-position, y-position, orientation]
# where x/y-position is its position in a given
# grid and orientation is 0-3 corresponding to 'up',
# 'left', 'down' or 'right'.
#
# Your task is to compute and return the car's optimal
# path to the position specified in `goal'; where
# the costs for each motion are as defined in `cost'.

# EXAMPLE INPUT:

# grid format:
#     0 = navigable space
#     1 = occupied space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]
goal = [2, 0] # final position
init = [4, 3, 0] # first 2 elements are coordinates, third is direction
cost = [2, 1, 20] # the cost field has 3 values: right turn, no turn, left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D() should return the array
# 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
#
# ----------


# there are four motion directions: up/left/down/right
# increasing the index in this array corresponds to
# a left turn. Decreasing is is a right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # do right
forward_name = ['up', 'left', 'down', 'right']

# the cost field has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']


# ----------------------------------------
# modify code below
# ----------------------------------------

def getNeighbors(node):
    currentNeighbor = []
    neighbors = []
    for i in range(len(forward)):
        currentNeighbor = [node[0] + forward[i][0], node[1] + forward[i][1]]
        if (((currentNeighbor[0] >= 0) and (currentNeighbor[0] < len(grid))) and \
           ((currentNeighbor[1] >= 0) and (currentNeighbor[1] < len(grid[i])))):
            neighbors.append(currentNeighbor)
    return neighbors

def filterNodes(nodes, nodesNotAvailable):
    filteredNodes = []
    for i in range(len(nodes)):
        if nodes[i] not in nodesNotAvailable:
            filteredNodes.append(nodes[i])
    return filteredNodes

value = [[[999 for row in range(len(grid[0]))] for col in range(len(grid))],
         [[999 for row in range(len(grid[0]))] for col in range(len(grid))],
         [[999 for row in range(len(grid[0]))] for col in range(len(grid))],
         [[999 for row in range(len(grid[0]))] for col in range(len(grid))]]
currentNode = goal
nodeNeighbors = []
blockedNodesList = []
openNodes = [goal]
def assignCostsToValue(robotOrientation):
    #Robot is facing North and forward motion is UP
    if (robotOrientation == 0):
        
    #For go straight
                value[0][nodeNeighbors[i][0]][nodeNeighbors[i][1]] = value[0][currentNode[0]][currentNode[1]] + cost[1]
                #For turn left and go straight
                value[1][nodeNeighbors[i][0]][nodeNeighbors[i][1]] = value[1][currentNode[0]][currentNode[1]] + cost[2]
                #For turn 180 degree and go down - Here we are making the assumption that a 180 degree turn will be made by 2 right turns
                value[2][nodeNeighbors[i][0]][nodeNeighbors[i][1]] = value[2][currentNode[0]][currentNode[1]] + 2*cost[0]
                #For turn right and go straight
                value[3][nodeNeighbors[i][0]][nodeNeighbors[i][1]] = value[3][currentNode[0]][currentNode[1]] + cost[0]
def compute_value(robotOrientation):
    #Value function is calculated based on current orientation of robot. Whenever the orientation is changed, value function needs to be recalculated
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
               blockedNodesList.append([i,j])
    value[0][goal[0]][goal[1]] = 0
    value[1][goal[0]][goal[1]] = 0
    value[2][goal[0]][goal[1]] = 0
    value[3][goal[0]][goal[1]] = 0
    while len(openNodes) != 0:
        currentNode = openNodes[0]
        nodeNeighbors = getNeighbors(currentNode)
        nodeNeighbors = filterNodes(nodeNeighbors, blockedNodesList)
        for i in range(len(nodeNeighbors)):
            if value[0][nodeNeighbors[i][0]][nodeNeighbors[i][1]] == 999:
                openNodes.append(nodeNeighbors[i])
                for direction in range(len(value)):
                    value[i][nodeNeighbors[i][0]][nodeNeighbors[i][1]] = value[0][currentNode[0]][currentNode[1]]
        openNodes.remove(currentNode)
    for i in range(len(value)):
        for j in range(len(value[i])):
            print value[i][j]
        print "================="
def sortKeyForAllDirections(neighborNode):
    directionalNeighborValues = []
    for i in range(len(value)):
        directionalNeighborValues.append(value[i][neighborNode[0]][neighborNode[1]])
    print sorted(directionalNeighborValues)
    return sorted(directionalNeighborValues)
def getDirectionSymbol(currentNode, nextNode):
    if currentNode[0] - nextNode[0] == 0:
        directionToTake = [0, nextNode[1] - currentNode[1]]
    elif currentNode[1] - nextNode[1] == 0:
        directionToTake = [nextNode[0] - currentNode[0], 0]
    else:
        print "Error in finding direction. Diagonal motion detected!"
    return forward_name[forward.index(directionToTake)]
def optimum_policy2D(robotDirection):
    policy2D = [[" " for col in range(len(grid[0]))] for row in range(len(grid))]
    policy2D[goal[0]][goal[1]] = "*"
    compute_value()
    assignCostsToValue(robotDirection)
    for i in range(len(value)):
        for j in range(len(value[i])):
            currentNode = [i,j]
            if currentNode != goal and value[0][currentNode[0]][currentNode[1]] != 999:
                nodeNeighbors = getNeighbors(currentNode)
                nodeNeighbors = sorted(nodeNeighbors, key=sortKeyForAllDirections)
                print currentNode
                print nodeNeighbors
                print "======================="
                policy2D[i][j] = getDirectionSymbol(currentNode, nodeNeighbors[0])
    for i in range(len(policy2D)):
        print policy2D[i]
    
    return policy2D # Make sure your function returns the expected grid.
#optimum_policy2D()
compute_value()
currentNode = [2,3]
nodeNeighbors = getNeighbors(currentNode)
print nodeNeighbors
nodeNeighbors = sorted(nodeNeighbors, key=sortKeyForAllDirections)
print nodeNeighbors
##policy = [[" " for col in range(len(grid[0]))] for row in range(len(grid))]
##policy[goal[0]][goal[1]] = "*"
##policy[3][3] = getDirectionSymbol(currentNode, nodeNeighbors[0])
##for i in range(len(policy)):
##    print policy[i]
