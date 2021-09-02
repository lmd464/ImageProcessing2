import numpy as np

src = np.array([[31, 2, 4, 33, 5, 36],
       [12, 26, 9, 10, 29, 25],
       [13, 17, 21, 22, 20, 18],
       [24, 23, 15, 16, 14, 19],
       [30, 8, 28, 27, 11, 7],
       [1, 35, 34, 3, 32, 6]])



def get_integral_image(src):
    assert len(src.shape) == 2
    h, w = src.shape
    dst = np.zeros(src.shape)
    ##############################
    # ToDo
    # dst는 integral image
    # dst 알아서 채우기
    ##############################
    for integral_i in range(h): # 0 ~ h - 1
        for integral_j in range(w):
            temp = src[0:integral_i + 1, 0:integral_j + 1] # 0-0 ~ 0-h-1
            dst[integral_i][integral_j] = np.sum(temp)

    return dst

dst = get_integral_image(src)
print()