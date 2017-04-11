import csv
import os

def delete_file(file_path):
    try:
        os.path.isfile(file_path)
        os.unlink(file_path)
    except Exception as e:
        print(e)

def load_lines(path):
    lines = []
    with open(path + "/driving_log_tobe_removed.csv") as datafile:
        reader = csv.reader(datafile)
        for line in reader:
            lines.append(line)
    return lines
def main():
    base_path = "D:/Udacity_SDCND/project-3-data/bridge-left-recovery"
    img_path = "/IMG/"
    lines = load_lines(base_path)
    center = lines[0][0]
    delete_file(base_path + img_path + center.replace("\\", "/").split("/")[-1])
    for line in lines:
        center, left, right = line[0], line[1], line[2]
        delete_file(base_path + img_path + center.replace("\\", "/").split("/")[-1])
        delete_file(base_path + img_path + left.replace("\\", "/").split("/")[-1])
        delete_file(base_path + img_path + right.replace("\\", "/").split("/")[-1])
    
main()
