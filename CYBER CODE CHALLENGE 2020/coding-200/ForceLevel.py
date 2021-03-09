from cv2 import cv2
import numpy as np
import subprocess


def f(x,y,rgb1,rgb2):
    r1,g1,b1 = rgb1
    r2,g2,b2 = rgb2
    xystr = str(x) + str(y)
    rgb1str = '{:0{}X}'.format(r1, 2) + '{:0{}X}'.format(g1, 2) + '{:0{}X}'.format(b1, 2)
    rgb2str = '{:0{}X}'.format(r2, 2) + '{:0{}X}'.format(g2, 2) + '{:0{}X}'.format(b2, 2)
    return xystr+rgb1str+rgb2str

for level in range(858,100000):
    image1 = cv2.imread(f'level_{level}.png')
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(f'lev3l_{level}.png')
    image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)

    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.flip(image2,0)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))
    if differences.shape not in [(3,1),(3,2),(3,3)]:
        image2 = cv2.rotate(image2,cv2.ROTATE_90_CLOCKWISE)
    differences = np.asarray(np.where(image1 != image2))


    if differences.shape not in [(3,1),(3,2),(3,3)]:
        print(level)
        print(differences)
        print(differences.shape)
        cv2.imshow('orig',image1)
        cv2.imshow('edit',image2)
        cv2.waitKey()
    pixeldiff = differences[[0,1],0]
    p1 = image1[pixeldiff[0],pixeldiff[1]]
    p2 = image2[pixeldiff[0],pixeldiff[1]]
    h,w,_ = image1.shape
    
    print('######')
    print(pixeldiff[1],pixeldiff[0])
    print(w-pixeldiff[1]-1,pixeldiff[0])
    print(pixeldiff[1],h-pixeldiff[0]-1)
    print(w-pixeldiff[1]-1,h-pixeldiff[0]-1)

    for pswd in [
        f(pixeldiff[1],pixeldiff[0],p1,p2),
        f(w-pixeldiff[1]-1,pixeldiff[0],p1,p2),
        f(pixeldiff[1],h-pixeldiff[0]-1,p1,p2),
        f(w-pixeldiff[1]-1,h-pixeldiff[0]-1,p1,p2),
    ]:
        #pswd = f(pixeldiff[1],pixeldiff[0],p1,p2)
        print(differences)
        print(pswd)
        decrypt = subprocess.Popen(['7z', 'e', f'level_{level+1}.7z', f'-p{pswd}'])
        decrypt.communicate()
        decrypt.wait()

    delete = subprocess.Popen(['rm',f'level_{level+1}.7z'])
    delete.communicate()
    delete.wait()



