from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import io
from convert import *


def settings(image, c=2, e=0.5):
    copy = ImageEnhance.Brightness(image).enhance(e)
    copy = ImageEnhance.Contrast(copy).enhance(c)
    return copy


def cmp_(x, y):
    return x[1] > y[1]


def histogram(image):
    y = image.histogram()
    MAX = []
    for i in range(256):
        if y[i] != 0:
            MAX.append((i, y[i]))
    MAX.sort(key=lambda x: x[1], reverse=True)
    print(MAX)
    return MAX


def showImg(arr):
    plt.imshow(arr, cmap='gray')
    arr = np.flip(arr, 0)
    plt.contour(arr, levels=[0], colors='white', origin='image')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    im.convert("L")
    im.show()
    buf.close()
    return im


def get_points(image):
    return (50, 70, 490, 370) # x1, y1, x2, y2


def detect(fname):
    im = init(fname)
    copy = im.crop(get_points(im))
    copy = settings(copy, 6, 0.5)
    while (histogram(copy)[0][0] != 255) and (histogram(copy)[0][0] != 0):
        copy = settings(copy, 1.2, 1)
    return copy

def light(copy, im):
    t = get_points(im)
    arr = np.array(copy)
    output_arr = np.array(im)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            output_arr[i + t[1]][j + t[0]][0] = min(255, int(output_arr[i + t[1] ][j + t[0] ][0]) + int(arr[i][j]))
            output_arr[i + t[1]][j + t[0]][1] = min(255, int(output_arr[i + t[1]][j + t[0]][1]) + int(arr[i][j]))

    im = Image.fromarray(output_arr)
    im.show()
    return im


if __name__ == "__main__":
    fname = "s.png"
    im = Image.open(fname).convert("RGB")
    #showImg(detect(fname))
    light(detect(fname), im).save("output1.png")