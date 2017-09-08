import matplotlib.pyplot as plt

def main():
    first_timestamp = -1
    x_axis = []
    y_axis_laser = []
    y_axis_radar = []
    laser_ninety_five_line = 5.991
    radar_ninety_five_line = 7.815
    laser_above_ninety_five_count = 0
    radar_above_ninety_five_count = 0
    file_handle = open("./../output-laser-radar-dataset-2.txt", "r")
    lines = file_handle.readlines()
    for line in lines:
        current_line_contents = line.split()
        if (current_line_contents[2] == "NIS"):
            continue
        if (first_timestamp == -1):
            first_timestamp = int(current_line_contents[1])
            x_axis.append(0)
        else:
            x_axis.append(int(current_line_contents[1]) - first_timestamp)
        if (int(current_line_contents[0]) == 0):
            y_axis_laser.append(float(current_line_contents[2]))
            if (float(current_line_contents[2]) > laser_ninety_five_line):
                laser_above_ninety_five_count += 1
        else:
            y_axis_radar.append(float(current_line_contents[2]))
            if (float(current_line_contents[2]) > radar_ninety_five_line):
                radar_above_ninety_five_count += 1

    plt.plot(y_axis_laser)
    plt.axhline(laser_ninety_five_line, color="black")
    plt.title("LASER NIS - " + str((laser_above_ninety_five_count*100/len(y_axis_laser))) + "% above allowed 95% line")
    plt.xlabel('Number of values')
    plt.ylabel('NIS')
    plt.show()
    plt.plot(y_axis_radar)
    plt.axhline(radar_ninety_five_line, color="black")
    plt.title("RADAR NIS - " + str((radar_above_ninety_five_count*100/len(y_axis_radar))) + "% above allowed 95% line")
    plt.xlabel('Number of values')
    plt.ylabel('NIS')
    plt.show()

if __name__ == "__main__":
   main()
