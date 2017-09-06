import matplotlib.pyplot as plt

def main():
    first_timestamp = -1
    x_axis = []
    y_axis = []
    file_handle = open("./../output.txt", "r")
    lines = file_handle.readlines()
    for line in lines:
        current_line_contents = line.split()
        if (current_line_contents[1] == "NIS"):
            continue
        if (first_timestamp == -1):
            first_timestamp = int(current_line_contents[0])
            x_axis.append(0)
        else:
            x_axis.append(int(current_line_contents[0]) - first_timestamp)
        y_axis.append(float(current_line_contents[1]))
    plt.plot(y_axis)
    plt.show()

if __name__ == "__main__":
   main()
