# -----------
# User Instructions:
# 
# Modify the function search() so that it returns
# a table of values called expand. This table
# will keep track of which step each node was
# expanded.
#
# For grading purposes, please leave the return
# statement at the bottom.
# ----------


grid = [[0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

cost = 1


# ----------------------------------------
# modify code below
# ----------------------------------------

#This code was written by udacity. Hence commenting this block of code.
##def search():
##    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
##    closed[init[0]][init[1]] = 1
##
##    x = init[0]
##    y = init[1]
##    g = 0
##
##    open = [[g, x, y]]
##
##    found = False  # flag that is set when search is complete
##    resign = False # flag set if we can't find expand
##
##    while not found and not resign:
##        if len(open) == 0:
##            resign = True
##        else:
##            open.sort()
##            open.reverse()
##            next = open.pop()
##            x = next[1]
##            y = next[2]
##            g = next[0]
##            
##            if x == goal[0] and y == goal[1]:
##                found = True
##            else:
##                for i in range(len(delta)):
##                    x2 = x + delta[i][0]
##                    y2 = y + delta[i][1]
##                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
##                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
##                            g2 = g + cost
##                            open.append([g2, x2, y2])
##                            closed[x2][y2] = 1
##    return expand #Leave this line for grading purposes!
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
    expansionGValue = 0
    currentNode = init
    visitedNodesList.append(init)

    expansionGrid = [[-1 for row in range(len(grid[0]))] \
                     for col in range(len(grid))]
    expansionGrid[0][0] = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                expansionGrid[i][j] = -1

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
                expansionGrid[openNodesList[i][0]][openNodesList[i][1]] = expansionGValue
                nodesTripletInPath.append([currentGValue] + openNodesList[i])

            nodesTripletInPath = sorted(nodesTripletInPath, key=lambda node: node[0])
        #If nodesTripletInPath is empty after finding openNodes, we can infer
        #that all nodes were visited but path was not found            
        if len(nodesTripletInPath) == 0:
            isPathNotFound = True
            break;
        for i in range(len(openNodesList)):
                expansionGrid[openNodesList[i][0]][openNodesList[i][1]] = expansionGValue
        
        currentNode = [nodesTripletInPath[0][1], nodesTripletInPath[0][2]]
        expansionGValue += 1
        currentGValue = nodesTripletInPath[0][0]
        visitedNodesList.append(init)
        nodesTripletInPath.remove(nodesTripletInPath[0])
    
##    if (isPathNotFound):
##        return "fail"
##    else:
##        return [currentGValue, currentNode[0], currentNode[1]]
    for i in range(len(expansionGrid)):
        print expansionGrid[i]
    return expansionGrid
search()
    




