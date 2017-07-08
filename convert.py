from PIL import Image, ImageOps
from numpy import array


def mk_arr(image):
    arr = array(image)
    return arr


def init(fname):
    im = Image.open(fname)
    if im.getbands()[0] != "L":
        im = im.convert("L")
    arr = mk_arr(im)
    #if arr[10][10] > 200:
        #im = ImageOps.invert(im)
    return im


if __name__ == "__main__":
    for i in range(3,50):
        try:
            a = init("Data/figure" + str(i) + "a.png")
            a.save("Data/" + str(i) + ".png")
        except Exception:
            print(i)
            continue
