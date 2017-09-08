import math
def main(observation, nearest_landmark, sigma_x=0.3, sigma_y=0.3):
    x = observation[0]
    y = observation[1]
    ux = nearest_landmark[0]
    uy = nearest_landmark[1]
    prob = ((1/(2*math.pi*sigma_x*sigma_y))*math.exp((-1)*(((x-ux)**2/(2*sigma_x**2)) + ((y-uy)**2/(2*sigma_y**2)))))
    return prob

#1 - Obs: (6,3), Nearest landmark: (5,3)
obs1 = (6, 3)
landmark1 = (5,3)
print(main(obs1, landmark1))

#2 - Obs: (2,2), Nearest landmark: (2,1)
obs2 = (2, 2)
landmark2 = (2,1)
print(main(obs2, landmark2))

#1 - Obs: (0,5), Nearest landmark: (2,1)
obs3 = (6, 3)
landmark3 = (2,1)
print(main(obs3, landmark3))
