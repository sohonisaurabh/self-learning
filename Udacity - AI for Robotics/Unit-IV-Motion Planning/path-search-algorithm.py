# ----------
# User Instructions:
# 
# Define a function, search() that takes no input
# and returns a list
# in the form of [optimal path length, x, y]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1] # Make sure that the goal definition stays in the function.
#goal = [0,1]

delta = [[-1, 0 ], # go up
        [ 0, -1], # go left
        [ 1, 0 ], # go down
        [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

cost = 1

def getOpenNodes(node, gridDim):
    openNodes = []
    for i in range(len(delta)):
        newNode = [node[0] + delta[i][0], node[1] + delta[i][1]]
        if ((newNode[0] >= 0 and newNode[0] < gridDim[0]) and \
            (newNode[1] >= 0 and newNode[1] < gridDim[1])):
            openNodes.append(newNode)
    return openNodes

def filterNodes(nodes, nodesNotAvailable):
    filteredNodes = []
    for i in range(len(nodes)):
        if nodes[i] not in nodesNotAvailable:
            filteredNodes.append(nodes[i])
    return filteredNodes
def search():
    # ----------------------------------------
    # insert code here and make sure it returns the appropriate result
    # ----------------------------------------
    openNodesList = []
    nodesTripletInPath = []
    blockedNodesList = []
    visitedNodesList = []
    isPathNotFound = False
    gridDim = [len(grid), len(grid[0])]
    currentGValue = 0
    currentNode = init
    visitedNodesList.append(init)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
               blockedNodesList.append([i,j])

    while currentNode != goal:
        openNodesList = getOpenNodes(currentNode, gridDim)
        openNodesList = filterNodes(openNodesList, blockedNodesList)
        openNodesList = filterNodes(openNodesList, visitedNodesList)

        if len(openNodesList) > 0:
            currentGValue += 1
            for i in range(len(openNodesList)):
                if openNodesList[i] not in visitedNodesList:
                    visitedNodesList.append(openNodesList[i])
            
            for i in range(len(openNodesList)):
                nodesTripletInPath.append([currentGValue] + openNodesList[i])

            nodesTripletInPath = sorted(nodesTripletInPath, key=lambda node: node[0])
        #If nodesTripletInPath is empty after finding openNodes, we can infer
        #that all nodes were visited but path was not found            
        if len(nodesTripletInPath) == 0:
            isPathNotFound = True
            break;
        
        currentNode = [nodesTripletInPath[0][1], nodesTripletInPath[0][2]]
        currentGValue = nodesTripletInPath[0][0]
        visitedNodesList.append(init)
        nodesTripletInPath.remove(nodesTripletInPath[0])
    
    if (isPathNotFound):
        return "fail"
    else:
        return [currentGValue, currentNode[0], currentNode[1]]
        
print search()
    
    




