# -----------
# User Instructions:
#
# Modify the the search function so that it returns
# a shortest path as follows:
# 
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left, 
# up, and down motions. NOTE: the 'v' should be 
# lowercase.
#
# Your function should be able to do this for any
# provided grid, not just the sample grid below.
# ----------


# Sample Test case
grid = [[0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 0, 1 ], # go right
         [ 1, 0 ]] # go down

delta_name = ['^', '<', '>', 'v']

cost = 1

# ----------------------------------------
# modify code below
# ----------------------------------------

#This search algorithm was written by udacity people. Below is my developed code.
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
##    found = False  # flag that is set when search is complet
##    resign = False # flag set if we can't find expand
##
##    while not found and not resign:
##        if len(open) == 0:
##            resign = True
##            return 'fail'
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
##    for i in range(len(expand)):
##        print expand[i]
##    return # make sure you return the shortest path.
class node:
    def __init__(self, nodeCoord=[0, 0]):
        self.coord = nodeCoord
        self.expandedBy = []
        self.expandedTo = []
         
    def setExpandedTo(self, nodeCoord):
        self.expandedTo.append(nodeCoord)

    def setExpandedFrom(self, nodeCoord):
        self.expandedBy = nodeCoord
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
def getDirectionSymbol(currentNode, nextNode):
    if currentNode[0] - nextNode[0] == 0:
        directionToTake = [0, currentNode[1] - nextNode[1]]
    elif currentNode[1] - nextNode[1] == 0:
        directionToTake = [currentNode[0] - nextNode[0], 0]
    else:
        print "Error in finding direction. Diagonal motion detected!"
    return delta_name[delta.index(directionToTake)]
def search():
    # ----------------------------------------
    # insert code here and make sure it returns the appropriate result
    # ----------------------------------------
    openNodesList = []
    nodesTripletInPath = []
    blockedNodesList = []
    visitedNodesList = []
    nodesData = []
    isPathNotFound = False
    gridDim = [len(grid), len(grid[0])]
    currentGValue = 0
    expansionGValue = 1
    currentNode = node(init)
    visitedNodesList.append(init)

    nodesData.append(currentNode)

    expansionGrid = [[-1 for row in range(len(grid[0]))] \
                     for col in range(len(grid))]
    expansionGrid[0][0] = 0
    pathGrid = [[" " for row in range(len(grid[0]))] \
                     for col in range(len(grid))]
    pathGrid[0][0] = " "
    pathGrid[goal[0]][goal[1]] = "*"
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                expansionGrid[i][j] = -1

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
               blockedNodesList.append([i,j])

    while currentNode.coord != goal:
        openNodesList = getOpenNodes(currentNode.coord, gridDim)
        openNodesList = filterNodes(openNodesList, blockedNodesList)
        openNodesList = filterNodes(openNodesList, visitedNodesList)
            

        for i in range(len(openNodesList)):
            currentNode.setExpandedTo(openNodesList[i])
        
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
        
        currentNode = node([nodesTripletInPath[0][1], nodesTripletInPath[0][2]])
        nodesData.append(currentNode)
        expansionGValue += 1
        currentGValue = nodesTripletInPath[0][0]
        nodesTripletInPath.remove(nodesTripletInPath[0])
    
    if (isPathNotFound):
        print "fail"
    else:
        #print [currentGValue, currentNode.coord[0], currentNode.coord[1]]
        backTraceNode = nodesData[len(nodesData)-1]
        while backTraceNode.coord != init:
            counter = len(nodesData) - 1
            while counter >= 0:
                currentNodeInCheck = nodesData[counter]
                if backTraceNode.coord in currentNodeInCheck.expandedTo:
                    pathGrid[currentNodeInCheck.coord[0]][currentNodeInCheck.coord[1]] = getDirectionSymbol(backTraceNode.coord, currentNodeInCheck.coord)
                    backTraceNode = currentNodeInCheck
                    break
                counter -= 1
        for j in range(len(pathGrid)):
            if j == 0 :
                print "["+str(pathGrid[j])+","
            elif j == len(pathGrid) - 1:
                print str(pathGrid[j])+"]"
            else:
                print str(pathGrid[j])+","
search()




