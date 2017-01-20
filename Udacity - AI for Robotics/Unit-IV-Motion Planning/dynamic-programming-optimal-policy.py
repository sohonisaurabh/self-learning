# ----------
# User Instructions:
# 
# Create a function optimum_policy() that returns
# a grid which shows the optimum policy for robot
# motion. This means there should be an optimum
# direction associated with each navigable cell.
# 
# un-navigable cells must contain an empty string
# WITH a space, as shown in the previous video.
# Don't forget to mark the goal with a '*'

# ----------

grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 0]]

init = [0, 0]
#goal = [len(grid)-1, len(grid[0])-1]
goal = [2,0]

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

cost_step = 1 # the cost associated with moving from a cell to an adjacent one.

# ----------------------------------------
# modify code below
# ----------------------------------------
policy = [[" " for col in range(len(grid[0]))] for row in range(len(grid))]
policy[goal[0]][goal[1]] = "*"
currentNode = []
def optimum_policy():
    value = [[99 for row in range(len(grid[0]))] for col in range(len(grid))]
    change = True

    while change:
        change = False

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if goal[0] == x and goal[1] == y:
                    if value[x][y] > 0:
                        value[x][y] = 0

                        change = True

                elif grid[x][y] == 0:
                    for a in range(len(delta)):
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]

                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                            v2 = value[x2][y2] + cost_step

                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
    for i in range(len(value)):
        for j in range(len(value[i])):
            currentNode = [i,j]
            if currentNode != goal and value[currentNode[0]][currentNode[1]] != 99:
                nodeNeighbors = getNeighbors(currentNode)
                nodeNeighbors = sorted(nodeNeighbors, key=lambda node: value[node[0]][node[1]])
                policy[i][j] = getDirectionSymbol(currentNode, nodeNeighbors[0])
    for i in range(len(policy)):
        print policy[i]
    return policy # Make sure your function returns the expected grid.
def getNeighbors(node):
    currentNeighbor = []
    neighbors = []
    for i in range(len(delta)):
        currentNeighbor = [node[0] + delta[i][0], node[1] + delta[i][1]]
        if (((currentNeighbor[0] >= 0) and (currentNeighbor[0] < len(grid))) and \
           ((currentNeighbor[1] >= 0) and (currentNeighbor[1] < len(grid[i])))):
            neighbors.append(currentNeighbor)
    return neighbors

def getDirectionSymbol(currentNode, nextNode):
    if currentNode[0] - nextNode[0] == 0:
        directionToTake = [0, nextNode[1] - currentNode[1]]
    elif currentNode[1] - nextNode[1] == 0:
        directionToTake = [nextNode[0] - currentNode[0], 0]
    else:
        print "Error in finding direction. Diagonal motion detected!"
    return delta_name[delta.index(directionToTake)]

optimum_policy()


