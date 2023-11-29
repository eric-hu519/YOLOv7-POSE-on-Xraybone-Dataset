import numpy as np
import os
import json
from tqdm import tqdm
import argparse
#import nbimporter
from format_conversion import *

parser = argparse.ArgumentParser()

parser.add_argument('--json_path', default=r'./coco_converter/annotations',
type=str,
help="input: coco format(json)")

parser.add_argument('--save_path', default=r'./final_dataset/labels', type=str,
help="specify where to save the output dir of labels")
parser.add_argument('--json_type', default=r'xRayBone_test.json', type=str,
help="import train/val or test .json file?")
parser.add_argument('--save_type',default=r'test',type=str,
                    help="export train/val or test .txt file")
parser.add_argument('--img_folder',default=r'images',
                    help="specify the folder for images")
parser.add_argument('--rename_img',default=r'False',
                    help="need to rename the images according to id?")
parser.add_argument('--display_flag',default=r'True',
                    help="display the images with annotations?")
parser.add_argument('--save_txt_flag',default=r'True')
arg = parser.parse_args(args=[])




cwd = os.getcwd()
json_file = arg.json_path# Annotation of COCO Object Instance type
save_type = arg.save_type
save_file = arg.save_path # saved path
rename_flag = bool(arg.rename_img)
img_folder = arg.img_folder
ana_txt_save_path = os.path.join(cwd,save_file,save_type)
json_type = arg.json_type
json_path = os.path.join(cwd, json_file)
file_path = os.path.join(json_path, json_type)
img_folder = os.path.join(cwd,img_folder,save_type)
conv = my_converter(file_path,save_type,ana_txt_save_path,img_folder,rename_flag)
conv.converter()
