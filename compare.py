import numpy as np
import os
import json
import matplotlib.pyplot as plt

cwd = os.getcwd()

c_net_path = os.path.join("records")

yolo_path = os.path.join("runs","detect","exp14","labels")

test_json = os.path.join("coco_converter","annotations","xRayBone_test.json")
lables = os.path.join("final_dataset","labels","test")


def get_Eular_dist(lable_results,c_net_results,yolo_results):
    lable_results = np.array(lable_results)
    c_net_results = np.array(c_net_results)
    yolo_results = np.array(yolo_results)

    c_net_dist = np.sqrt(np.sum((c_net_results - lable_results)**2, axis=-1))
    yolo_net_dist = np.sqrt(np.sum((yolo_results - lable_results)**2, axis=-1))
    
    
    print("the mean distance of c_net is")
    print(np.mean(c_net_dist))
    print("the mean distance of yolo is ")
    print(np.mean(yolo_net_dist))
    return c_net_dist, yolo_net_dist

labels_result_list = []
yolo_result_list = []
c_net_result_list = []

def get_class_dist(A_test,s1_test,f1_test,f2_test,A_list,s1_list,f1_list,f2_list):
    A_test = np.array(A_test)
    s1_test = np.array(s1_test)
    f1_test = np.array(f1_test)
    f2_test = np.array(f2_test)

    A_list = np.array(A_list)
    s1_list = np.array(s1_list)
    f1_list = np.array(f1_list)
    f2_list = np.array(f2_list)
    print(A_list.shape)
    print(A_test.shape)
    A_dist = np.sqrt(np.sum((A_list - A_test)**2, axis=-1))
    s1_dist = np.sqrt(np.sum((s1_list - s1_test)**2, axis=-1))
    f1_dist = np.sqrt(np.sum((f1_list - f1_test)**2, axis=-1))
    f2_dist = np.sqrt(np.sum((f2_list - f2_test)**2, axis=-1))
        
    print("the mean distance of A is")
    print(np.mean(A_dist))
    print("the mean distance of s1 is ")
    print(np.mean(s1_dist))

    print("the mean distance of f1 is ")
    print(np.mean(f1_dist))
    print("the mean distance of f2 is ")
    print(np.mean(f2_dist))

A_test = []
s1_test = []
f1_test = []
f2_test = []
def get_anno(test_json):
    lables = []
    data = json.load(open(test_json, 'r'))
    last_img_id = -1
    for annotations in data['annotations']:
    # filename = annotations["file_name"]
        img_id = annotations["image_id"]
        if img_id != last_img_id:
            for images in data['images']:
                if images["id"] == img_id:
                    img_width = images["width"]
                    img_height = images["height"]
                    test_file_name = images["file_name"]
                    test_file_name, tail= test_file_name.split('.')
                #if  annotations["keypoints"] != None:
            keypoints = annotations["keypoints"]
            cur_test = annotations["keypoints"]
            keypoints = keypoints[:2]
            A_test.append(cur_test[:2])
            anno_count = 0 #reset anno count
        bbox = annotations["bbox"]
        bbox_center = bbox_converter(bbox)
        if anno_count == 0:
            s1_test.append(bbox_center)
        if anno_count == 1:
            f1_test.append(bbox_center)
        if anno_count == 2:
            f2_test.append(bbox_center)
        keypoints+= bbox_center
        last_img_id = img_id
        anno_count += 1
        if anno_count == 3 :
        #gather all 3 annotations for one same img_id
            
            lables.append(keypoints)
            annotations = []#reset
            keypoints = []
    return lables

def get_img():
    data = json.load(open(test_json, 'r'))
    for images in data['images']:
        img_width = images["width"]
        img_height = images["height"]
        test_file_name = images["file_name"]
        img_id = images["id"]
        test_file_name, tail= test_file_name.split('.')
        #lables_results = get_lable_result(img_id)
        yolo_results = get_yolo_result(img_id,img_width,img_height)
        c_net_results = get_c_net_result(test_file_name,img_width,img_height)
        #labels_result_list.append(lables_results)
        yolo_result_list.append(yolo_results)
        c_net_result_list.append(c_net_results)
    labels_result_list = get_anno(test_json)
    c_net_dist,yolo_net_dist = get_Eular_dist(labels_result_list,c_net_result_list,yolo_result_list)
    get_class_dist(A_test,s1_test,f1_test,f2_test,A_list,s1_list,fh1_list,fh2_list)
    visulization(yolo_net_dist,c_net_dist)
    
A_list = []
s1_list = []
fh1_list = []
fh2_list = []
def get_yolo_result(img_id,img_width,img_height):
        yolo_file = os.path.join(yolo_path, str(img_id)+".txt")
        yolo_file = open(yolo_file,'r')
        yolo_result = yolo_file.readline().split()
        yolo_result = [float(x) for x in yolo_result]
        A = yolo_result[0:2]
        s1 = yolo_result[2:4]
        fh1 = yolo_result[4:6]
        fh2 = yolo_result[6:8]
        A_list.append(A)
        s1_list.append(s1)
        fh1_list.append(fh1)
        fh2_list.append(fh2)
        #A = val_norm(A,img_width,img_height)
        #s1 = val_norm(s1,img_width,img_height)
        #fh1 = val_norm(fh1,img_width,img_height)
        #fh2 = val_norm(fh2,img_width,img_height)
        yolo_result = A+s1+fh1+fh2
        return yolo_result

def get_lable_result(img_id):
        lables_file = os.path.join(lables,str(img_id)+".txt")
        lables_file = open(lables_file,'r')
        lables_result = lables_file.readline().split()
        lables_result = [float(x) for x in lables_result]
        A = lables_result[5:7]
        s1 = lables_result[8:10]
        fh1 = lables_result[11:13]
        fh2 = lables_result[14:16]
        lables_result = A+s1+fh1+fh2
        return lables_result

def get_c_net_result(test_file_name,img_width,img_height):
        c_net_record = os.path.join(c_net_path,test_file_name+".json")#open the corrspopnding results
        c_net_data = json.load(open(c_net_record,'r'))
        value_1 = c_net_data['1']
        value_2 = c_net_data['2']
        value_3 = c_net_data['3']
        A = value_1[-2:]
        s1 = bbox_converter(value_1[:4])
        fh1 = bbox_converter(value_2[:4])
        fh2 = bbox_converter(value_3[:4])
        #A = val_norm(A,img_width,img_height)
        #s1 = val_norm(s1,img_width,img_height)
        #fh1 = val_norm(fh1,img_width,img_height)
        #fh2 = val_norm(fh2,img_width,img_height)
        c_net_result=A+s1+fh1+fh2
        return c_net_result


def bbox_converter(bbox):
        x_min, y_min, bb_width, bb_height = bbox
        x_max = x_min + bb_width
        y_max = y_min + bb_height
        # compute center coordinates
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bbox_center = [center_x, center_y]
        return bbox_center

def val_norm(value,img_width,img_hight):
        accuracy = 10 ** 6 
        value[0] = int((value[0] / img_width) * accuracy) / accuracy
        value[1] = int((value[1] / img_hight) * accuracy) / accuracy
        return value


def visulization(yolo_dist,c_net_dist):
    model1_distances =yolo_dist
    model2_distances = c_net_dist
    # Create a scatter plot of the distances for both models
    fig, ax = plt.subplots()
    ax.scatter(model1_distances, model2_distances, c=model1_distances-model2_distances, cmap='coolwarm')
    ax.set_xlabel('Distances for yolo_dist')
    ax.set_ylabel('Distances for c_net_dist')
    ax.set_title('Comparison of Model Accuracy')
    plt.show()

    # Create a histogram of the distances for both models
    fig, ax = plt.subplots()
    ax.hist(model1_distances, alpha=0.5, label='Model 1')
    ax.hist(model2_distances, alpha=0.5, label='Model 2')
    ax.set_xlabel('Distances')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Model Accuracy')
    ax.legend()
    plt.show()




get_img()