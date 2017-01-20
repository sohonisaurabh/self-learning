colors = [['red', 'green', 'green', 'red' , 'red'],
          ['red', 'red', 'green', 'red', 'red'],
          ['red', 'red', 'green', 'green', 'red'],
          ['red', 'red', 'red', 'red', 'red']]

measurements = ['green', 'green', 'green' ,'green', 'green']


motions = [[0,0],[0,1],[1,0],[1,0],[0,1]]

sensor_right = 0.7

p_move = 0.8

sensor_wrong = 1 - sensor_right

p_move_overshoot = (1.0-p_move)/2.0
p_move_undershoot = p_move_overshoot
p_stay = 1 - p_move

def createCopy(originalArray):
    tempProb = []
    for i in range(len(originalArray)):
        tempProb.append([])
        for j in range(len(originalArray[i])):
            tempProb[i].append(originalArray[i][j])
    return tempProb
def calculateNormalizeWeight(list):
    weight = 0.0
    for i in range(len(list)):
        weight += sum(list[i])
    return weight
def normalize(probArray, weight):
    for i in range(len(probArray)):
        for j in range(len(probArray[i])):
            probArray[i][j] /= weight
def twoDMove(belief, motionSpec):
    tempBelief = [[0 for col in range(len(p[0]))] for row in range(len(p))]
    motionInX = motionSpec[0]
    motionInY = motionSpec[1]
    for i in range(len(p)):
        for j in range(len(p[i])):
            #Code with overshoot and undershoot
            #tempBelief[i][j] += p_move*p[(i-motionInX)%len(p)]\
             #              [(j-motionInY)%len(p[i])]
            #tempBelief[i][j] += p_move_overshoot*\
             #              p[(i-motionInX-(motionInX*1))%len(p)]\
              #             [(j-motionInY-(motionInY*1))%len(p[i])]
            #tempBelief[i][j] += p_move_undershoot*\
             #              p[(i-motionInX+(motionInX*1))%len(p)]\
              #             [(j-motionInY+(motionInY*1))%len(p[i])]
            #Code with move and stay
            tempBelief[i][j] += p_move*p[(i-motionInX)%len(p)]\
                           [(j-motionInY)%len(p[i])] + p_stay*p[i][j]
    return tempBelief
def twoDSense(priorBelief, measurement):
    posteriorBelief = [[0 for col in range(len(p[0]))] for row in range(len(p))]
    for row in range(len(colors)):
        for col in range(len(colors[row])):
            if measurement == colors[row][col]:
                posteriorBelief[row][col] = p[row][col]*sensor_right
            else:
                posteriorBelief[row][col] = p[row][col]*sensor_wrong
    normalizer = calculateNormalizeWeight(posteriorBelief)
    normalize(posteriorBelief, normalizer)
    return posteriorBelief
def show(p):
    for i in range(len(p)):
        print p[i]

#DO NOT USE IMPORT
#ENTER CODE BELOW HERE
#ANY CODE ABOVE WILL CAUSE
#HOMEWORK TO BE GRADED
#INCORRECT

totalCells = len(colors) * len(colors[0])
#Initializing to uniform distribution
p = [[(1.0/totalCells) for col in range(len(colors[0]))] for row in range (len(colors))]
for i in range(len(motions)):
    p = twoDMove(p, motions[i])
    p = twoDSense(p, measurements[i])

#Test data goes here
#p = [[0.1, 0.05, 0.05, 0.1, 0.05],
 #    [0.025, 0.05, 0.05, 0.05, 0.025],
  #   [0.05, 0.025, 0.05, 0.05, 0.05],
   #  [0.05, 0.05, 0.05, 0.025, 0.05]]
#p = [[0, 1, 0, 0, 0],
 #    [1, 0, 0, 0, 0],
  #   [0, 0, 1, 0, 0],
   #  [0, 0, 0, 1, 0]]

#Your probability array must be printed 
#with the following code.

show(p)




