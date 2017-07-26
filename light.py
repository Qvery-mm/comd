
import numpy as np
from PIL import Image
from numpy import array
import cv2
import json
from PyQt5 import QtWidgets as QW



black = 0
FULLSCREEN = True

def cvt(CNT, data):
    for i in CNT:
        i.write_message(data)


def binary(img):
    #img = cv2.imread(fname, 0)
    #img = cv2.resize(img, (512, 512))
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

def contour(original, arr):
    print(arr)
    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            original[i][j]*=arr[i][j]
    return original


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
    return maximum[0:5]

def work(i):
    global black
    # print('open image')
    # print()
    # print(i)
    # original = cv2.imread(i, 0)
    # original = cv2.resize(original, (512, 512))
    red = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)

    img = binary(i)
    # img.show()
    img = array(img)
    cv2.waitKey(10)
    ##select colour
    if img[img.shape[0] - 1][img.shape[1] // 2] == 0:

        BASECOLOR = 255
    else:
        BASECOLOR = 0

    for j in range(img.shape[0]):
        cv2.waitKey(10)
        y = [img[j][m] for m in range(img.shape[1])]
        for k in check(y, BASECOLOR):
            for x in range(k[1], k[2]):
                img[j][x] = 1
                red[j][x][0] = 255
                red[j][x][2] = 255
        if (j % 2) == 0:
            # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", red)
            continue

            cv2.waitKey(5)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] != 1:
                img[x][y] = 0
    black = img
    return red

    #contour(original, img).save(dir + "ANS/" + i)

    #cv2.waitKey(0)
    #contour(original, img).show()








def scan(img):
    cascadePath = "nodules6.0.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    ans = []

    for face in faces:
        face[2] += face[0]
        face[3] += face[1]
        ans.append(face)


    return ans

def qhist(self, im, clr=False):
    if clr: return cvt(self, json.dumps({"type": "histogram", "stat": [0 for i in range(256)]}))

    q = np.array(im.histogram())
    q = 100*q/max(q)

    cvt(self, json.dumps({"type": "histogram", "stat": list(map(int, q))}))

def ren(self, img):
    global black
    i = 0
    while i < 2.5:
        i = i + 0.1
        clahe = cv2.createCLAHE(clipLimit= i, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        cl1 = np.array(cl1)
        qhist(self, Image.fromarray(cl1))

        #print('@@@', i)
        if i <= 2:
            cv2.waitKey(10)  # 70
            cv2.imshow("window", cl1)
        elif i > 1:
            cv2.waitKey(5)
            positions = scan(cl1)

            cl1 = Image.fromarray(cl1)

            cl1 = cl1.convert("P")
            cl12 = Image.new("L", cl1.size, 255)

            cl1 = cl1.convert("1")

            cl1 = cl1.convert("L")

            cl1 = np.array(cl1)
            # qhist(self, Image.fromarray(cl1))
            cv2.imshow("window", cl1)
            cvt(self, json.dumps({"type": "processing", "stat": 10 * i}))


    cl1 = work(cl1)



    pos_to = []


    for pos in positions * 5:
        base_cli = cl1.copy()
        cv2.rectangle(base_cli, (pos[0], pos[1]), (pos[2], pos[3]), (200, 400, 0), 1)
        cv2.imshow("window", base_cli)
        qhist(self, Image.fromarray(cl1))
        cv2.waitKey(50)
    cvt(self, json.dumps({"type": "processing", "stat": 50}))


    cl1 = contour(img, black)
    cv2.imshow("window", cl1)
    cv2.waitKey(3000)

    # PIXELS

    cl1 = np.array(cl1)
    shag = 2
    summ = 0
    summ2 = 0
    summ1 = 0
    c = 0
    l = []
    x, y = cl1.shape

    for k in range(1, 4):
        for i in range(x // k):
            for j in range(y // k):
                #             time.sleep(1)
                for p in range(k * i, k * (i + 1)):
                    for p1 in range(k * j, k * (j + 1)):
                        #                     print(p, p1)
                        summ += cl1[p][p1]
                        c += 1
                        #             print("     ")
                    av = summ // c
                #             print(summ,av,c)
                summ, summ1, summ2, c = 0, 0, 0, 0
                for p in range(k * i, k * (i + 1)):
                    for p1 in range(k * j, k * (j + 1)):
                        cl1[p][p1] = av

        # cl1 = np.array(cl1)
        cv2.imshow("window", cl1)
        qhist(self, Image.fromarray(cl1))
        cv2.waitKey(1000)
        cvt(self, json.dumps({"type": "processing", "stat": max(50, min(100, k * 10 + 50))}))
        # if time.sleep(2):
        #     return cl1

    return cl1


# Create a black image, a window and bind the function to window
def kostil(self, fname):
    img = cv2.imread(fname, 0)

    if FULLSCREEN:
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        app = QW.QApplication([])
        sizeObject = QW.QDesktopWidget().screenGeometry(-1)
        lenght = sizeObject.height()
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window", lenght, lenght)
    #cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



    #   print('PS: start')

    res = ren(self, img)
    qhist(self, Image.fromarray(img))
    cv2.destroyAllWindows()

# while(1):
#     cv2.imshow("window", res)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break






