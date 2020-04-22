#
# src = "/home/lzm/Disk2/work_dl/Pytorch_refinedet/Refinedet_Pytorch-res86/in.txt"
# dst = open("/home/lzm/Disk2/work_dl/Pytorch_refinedet/Refinedet_Pytorch-res86/out.txt", 'w')
#
#
# output = []
# for i, line in enumerate(open(src).readlines()):
#     name = line.split('\n')[0]
#     info = "print("+"'"+str(i)+":"+name+"'"+","+name+".shape)" +"\n"
#     output.append(info)
#
# dst.writelines(output)

import os
src = "/home/lzm/Disk2/work_dl/Pytorch_refinedet/Data_dir/Test_model/RefineDet320_VOC_140000_withhead"
dirlist = os.listdir(src)
dirlist.sort()

for dirs in dirlist:
    dirA = os.path.join(src, dirs)
    listA = os.listdir(dirA)
    listA.sort()
    print(dirs, ":")
    for i in listA:
        print(i, ":", len(os.listdir(os.path.join(dirA, i))))
