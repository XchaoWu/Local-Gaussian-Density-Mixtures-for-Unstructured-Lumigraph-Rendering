import os,sys,cv2 
from glob import glob 

data_dir = sys.argv[1]
file_list = glob(os.path.join(data_dir, "*.png"))

file_list.sort(key = lambda x:  int(os.path.splitext(os.path.basename(x))[0]) )


for file in file_list:
    img = cv2.imread(file)
    index = int(os.path.splitext(os.path.basename(file))[0])
    out_path = os.path.join(data_dir, "%08d"%(index) + ".jpg")
    print(out_path)
    cv2.imwrite(out_path, img)
