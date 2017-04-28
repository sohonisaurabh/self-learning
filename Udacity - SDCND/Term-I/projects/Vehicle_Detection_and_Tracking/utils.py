import glob
import os

def rename_file(src_file_path, dst_file_path):
    print(src_file_path, "-------->>>", dst_file_path)
    os.rename(src_file_path, dst_file_path)
    
def main():
    base_path = "complete-dataset/vehicles/image*.png"
    images_names = glob.glob(base_path)
    for img_name in images_names:
        img_name = img_name.replace("\\", "/")
        complete_path = img_name.split("/")
        new_img_name = "/".join(complete_path[0: len(complete_path) - 1]) + "/gtiright-" + complete_path[-1]
        rename_file(img_name, new_img_name)
    
main()
