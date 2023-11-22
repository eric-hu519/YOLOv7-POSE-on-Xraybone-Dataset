import os
def get_kpt_coor(coor_list):
    if len(coor_list) == 8: #make sure the list has coordinates for 4 points
        f = open(os.path.join("key point coordinates"+".temp"),'w')
        for item in coor_list:
            f.write(str(item) + ' ')
        f.close