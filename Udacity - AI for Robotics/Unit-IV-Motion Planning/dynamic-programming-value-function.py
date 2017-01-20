# ----------
# User Instructions:
# 
# Create a function compute_value() which returns
# a grid of values. Value is defined as the minimum
# number of moves required to get from a cell to the
# goal. 
#
# If it is impossible to reach the goal from a cell
# you should assign that cell a value of 99.

# ----------

grid = [[0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
#goal = [2,2]

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

cost_step = 1 # the cost associated with moving from a cell to an adjacent one.

# ----------------------------------------
# insert code below
# ----------------------------------------

def getNeighbors(node):
    currentNeighbor = []
    neighbors = []
    for i in range(len(delta)):
        currentNeighbor = [node[0] + delta[i][0], node[1] + delta[i][1]]
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

value = [[99 for row in range(len(grid[0]))] for col in range(len(grid))]
currentNode = goal
nodeNeighbors = []
blockedNodesList = []
openNodes = [goal]
def compute_value():
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
               blockedNodesList.append([i,j])
    value[goal[0]][goal[1]] = 0
    while len(openNodes) != 0:
        currentNode = openNodes[0]
        nodeNeighbors = getNeighbors(currentNode)
        nodeNeighbors = filterNodes(nodeNeighbors, blockedNodesList)
        for i in range(len(nodeNeighbors)):
            if value[nodeNeighbors[i][0]][nodeNeighbors[i][1]] == 99:
                openNodes.append(nodeNeighbors[i])
                value[nodeNeighbors[i][0]][nodeNeighbors[i][1]] = value[currentNode[0]][currentNode[1]] + cost_step
        openNodes.remove(currentNode)
    for i in range(len(value)):
        print value[i]
        

    return value #make sure your function returns a grid of values as demonstrated in the previous video.
compute_value()


