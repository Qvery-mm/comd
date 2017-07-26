import cv2
import os
import queue
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from numpy import array


def binary(fname):
    img = cv2.imread(fname, 0)
    img = cv2.resize(img, (512, 512))
    img = cv2.equalizeHist(img)
    im = Image.fromarray(img)
    im = im.convert("P")
    im = im.convert("1")
    im = im.convert("L")
    return im


def summ(m):
    res = 0
    for i in m:
        res+=i[0]
    return res


def check(line, color):
    barrier = 20
    maximum = []
    i = 0
    while i < len(line):
        if line[i] == color:
            j = i
            while j < len(line) and line[j] == color:
                j+=1
            if i > barrier and j < len(line) - barrier and summ(maximum) < (len(line) / 1.5):
                maximum.append((j - i, i, j))
            i = j
        i+=1
    maximum.sort(reverse=True)
    return maximum[0:3]

def bfs(arr, p, colour):
    a = arr.shape
    sqr = 0
    q = queue.Queue()
    q.put(p)
    while not q.empty():
        x, y = q.get()
        if 0 <= x and x < a[0] and 0 <= y and y < a[1]:
            if arr[x][y] == colour:
                arr[x][y] = colour - 1
                sqr += 1
                q.put((x + 1, y))
                q.put((x - 1, y))
                q.put((x, y + 1))
                q.put((x, y - 1))
    return sqr

def contour(original, arr):

    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            original[i][j]*=arr[i][j]
    #im = Image.fromarray(original, mode="L")
    return im



def work(i):
    print(i)
    original = cv2.imread(i, 0)
    original = cv2.resize(original, (512, 512))
    red = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    img = binary(i)
    #img.show()
    img = array(img)

    ##select colour
    if img[510][256] == 0:
        BASECOLOR = 255
    else:
        BASECOLOR = 0



    for j in range(img.shape[0]):
        y = [img[j][m] for m in range(img.shape[1])]
        for k in check(y, BASECOLOR):
            for x in range(k[1], k[2]):
                img[j][x] = 1
                red[j][x][2] = 255
                red[j][x][1] = 255
        if (j%2) == 0:
            cv2.imshow("res", red)
            cv2.waitKey(1)
    #cv2.destroyWindow("res")

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] != 1:
                img[x][y] = 0

    #contour(original, img).save(dir + "ANS/" + i)

    #cv2.waitKey(0)
    cv2.imshow("res", contour(original, img))
    cv2.waitKey(0)
if __name__ == "__main__":
    work("C:/Users/User/Desktop/new_nodule_Alexandr/61.png")