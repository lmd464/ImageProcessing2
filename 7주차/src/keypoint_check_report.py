import numpy as np
import cv2
import random

def feature_matching(img1, img2, RANSAC=False, threshold = 300, keypoint_num = None, iter_num = 500, threshold_distance = 10):
    '''
    #ToDo
    #바뀐것은 return dst -> return dst, M 이것밖에 없습니다. 6주차 과제 완성한 내용을 그대로 복붙해주세요
    '''

    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    distance = []
    for idx_1, des_1 in enumerate(des1):
        dist = []
        for idx_2, des_2 in enumerate(des2):
            dist.append(L2_distance(des_1, des_2))

        distance.append(dist)

    distance = np.array(distance)

    min_dist_idx = np.argmin(distance, axis=1)
    min_dist_value = np.min(distance, axis=1)

    points = []
    for idx, point in enumerate(kp1):
        if min_dist_value[idx] >= threshold:
            continue

        x1, y1 = point.pt
        x2, y2 = kp2[min_dist_idx[idx]].pt

        x1 = int(np.round(x1))
        y1 = int(np.round(y1))

        x2 = int(np.round(x2))
        y2 = int(np.round(y2))
        points.append([(x1, y1), (x2, y2)])


    # no RANSAC
    if not RANSAC:

        A = []
        B = []
        for idx, point in enumerate(points):
            '''
            #ToDo
            #A, B 완성
            # A.append(???) 이런식으로 할 수 있음
            # 결과만 잘 나오면 다른방법으로 해도 상관없음
            '''
            A.append( [point[0][0], point[0][1], 1, 0, 0, 0] )
            A.append( [0, 0, 0, point[0][0], point[0][1], 1] )
            B.append( [point[1][0]] )
            B.append( [point[1][1]] )


        A = np.array(A)
        B = np.array(B)

        '''
        #ToDo
        #X 완성
        #np.linalg.inv(V) : V의 역행렬 구하는것
        #np.dot(V1, V2) : V1과 V2의 행렬곱
        # V1.T : V1의 transpose
        '''
        atainv = np.linalg.inv(np.dot(np.transpose(A), A))
        atb = np.dot(np.transpose(A), B)

        X = np.dot(atainv, atb)


        '''
        # ToDo
        # 위에서 구한 X를 이용하여 M 완성
        '''
        M = [ [X[0][0], X[1][0], X[2][0]],
              [X[3][0], X[4][0], X[5][0]],
              [0, 0, 1]         ]

        M = np.array(M)
        M_ = np.linalg.inv(M)


        '''
        # ToDo
        # backward 방식으로 dst완성
        '''
        #Backward 방식 : bilinear
        src = img1  # 원본
        h, w = img1.shape[:2]

        h_, w_ = img2.shape[:2]
        dst = np.zeros((h_, w_, 3))

        for row in range(h_):
            for col in range(w_):
                # bilinear
                vec = np.dot(M_, np.array([[col, row, 1]]).T)
                c = vec[0, 0]
                r = vec[1, 0]
                c_left = int(c)
                c_right = min(int(c + 1), w - 1)
                r_top = int(r)
                r_bottom = min(int(r + 1), h - 1)
                s = c - c_left
                t = r - r_top

                if 0 <= r_bottom and r_bottom < h and \
                        0 <= r_top and r_top < h and \
                        0 <= c_right and c_right < w and \
                        0 <= c_left and c_left < w:
                    intensity = (1 - s) * (1 - t) * src[r_top, c_left] \
                                + s * (1 - t) * src[r_top, c_right] \
                                + (1 - s) * t * src[r_bottom, c_left] \
                                + s * t * src[r_bottom, c_right]
                else:
                    intensity = 0

                dst[row, col] = intensity

        dst = dst.astype(np.uint8)


    #use RANSAAC
    else:
        points_shuffle = points.copy()

        inliers = []
        M_list = []
        for i in range(iter_num):
            random.shuffle(points_shuffle)
            three_points = points_shuffle[:3]

            A = []
            B = []
            #3개의 point만 가지고 M 구하기
            for idx, point in enumerate(three_points):
                '''
                #ToDo
                #A, B 완성
                # A.append(???) 이런식으로 할 수 있음
                # 결과만 잘 나오면 다른방법으로 해도 상관없음
                '''
                A.append([point[0][0], point[0][1], 1, 0, 0, 0])
                A.append([0, 0, 0, point[0][0], point[0][1], 1])
                B.append([point[1][0]])
                B.append([point[1][1]])

            A = np.array(A)
            B = np.array(B)
            try:
                '''
                #ToDo
                #X 완성
                #np.linalg.inv(V) : V의 역행렬 구하는것
                #np.dot(V1, V2) : V1과 V2의 행렬곱
                # V1.T : V1의 transpose 단, type이 np.array일때만 가능. type이 list일때는 안됨
                '''
                atainv = np.linalg.inv(np.dot(A.T, A))
                atb = np.dot(A.T, B)

                X = np.dot(atainv, atb)

            except:
                #print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
                continue

            '''
            # ToDo
            # 위에서 구한 X를 이용하여 M 완성
            '''
            M = [[X[0][0], X[1][0], X[2][0]],
                 [X[3][0], X[4][0], X[5][0]],
                 [0, 0, 1]]

            M_list.append(M)

            count_inliers = 0
            for idx, point in enumerate(points):
                '''
                # ToDo
                # 위에서 구한 M으로(3개의 point로 만든 M) 모든 point들에 대하여 예상 point 구하기
                # 구해진 예상 point와 실제 point간의 L2 distance 를 구해서 threshold_distance보다 작은 값이 있는 경우 inlier로 판단
                '''
                # M으로 구한 point
                point_predict = np.dot( M, np.array([point[0][0], point[0][1], 1]) )

                # 실제 point
                point_real = np.array([point[1][0], point[1][1], 1])


                if L2_distance(point_predict, point_real) < threshold_distance:
                    count_inliers += 1

            inliers.append(count_inliers)

        inliers = np.array(inliers)
        max_inliers_idx = np.argmax(inliers)

        best_M = np.array(M_list[max_inliers_idx])

        M = best_M
        M_ = np.linalg.inv(M)


        '''
        # ToDo
        # backward 방식으로 dst완성
        '''
        # Backward 방식 : bilinear
        src = img1  # 원본
        h, w = img1.shape[:2]

        h_, w_ = img2.shape[:2]
        dst = np.zeros((h_, w_, 3))

        for row in range(h_):
            for col in range(w_):
                # bilinear
                vec = np.dot(M_, np.array([[col, row, 1]]).T)
                c = vec[0, 0]
                r = vec[1, 0]
                c_left = int(c)
                c_right = min(int(c + 1), w - 1)
                r_top = int(r)
                r_bottom = min(int(r + 1), h - 1)
                s = c - c_left
                t = r - r_top

                if 0 <= r_bottom and r_bottom < h and \
                        0 <= r_top and r_top < h and \
                        0 <= c_right and c_right < w and \
                        0 <= c_left and c_left < w:
                    intensity = (1 - s) * (1 - t) * src[r_top, c_left] \
                                + s * (1 - t) * src[r_top, c_right] \
                                + (1 - s) * t * src[r_bottom, c_left] \
                                + s * t * src[r_bottom, c_right]

                else:
                    intensity = 0

                dst[row, col] = intensity

            dst = dst.astype(np.uint8)

    return dst, M


def L2_distance(vector1, vector2):
    '''
    #ToDo
    #6주차의 내용을 그대로 복붙해주세요
    '''
    distance = np.sqrt(np.sum((vector1 - vector2) ** 2))

    return distance




#실습 때 했던 코드입니다.
def scaling_test(src):
    h, w = src.shape[:2]
    rate = 2
    dst_for = np.zeros((int(np.round(h*rate)), int(np.round(w*rate)), 3))
    dst_back_bilinear = np.zeros((int(np.round(h*rate)), int(np.round(w*rate)), 3))
    M = np.array([[rate, 0, 0],
                  [0, rate, 0],
                  [0, 0, 1]])

    #FORWARD
    h_, w_ = dst_for.shape[:2]
    count = dst_for.copy()
    for row in range(h):
        for col in range(w):
            '''
            #ToDo
            #과제에서 사용하진 않지만 완성해주세요
            #실습을 참고해서 완성해주세요
            '''
            vec = np.dot(M, np.array( [[col, row, 1]] ).T)
            x = vec[0, 0]
            y = vec[1, 0]
            x1 = int(np.floor(x))
            x2 = int(np.ceil(x))
            y1 = int(np.floor(y))
            y2 = int(np.ceil(y))

            points_list = [(y1, x1), (y1, x2), (y2, x1), (y2, x2)]
            points = set(points_list)   # 중복제거

            for (row_, col_) in points:
                dst_for[min(row_, h_-1), min(col_, w_-1)] += src[row, col]
                count[min(row_, h_-1), min(col_, w_-1)] += 1

    dst_for = (dst_for / count).astype(np.uint8)


    #M 역행렬
    M_ = np.linalg.inv(M)
    print('M')
    print(M)
    print('M 역행렬')
    print(M_)
    h_, w_ = dst_back_bilinear.shape[:2]

    #BACKWARD
    for row_ in range(h_):
        for col_ in range(w_):
            '''
            #ToDo
            #bilinear
            #실습을 참고해서 완성해주세요
            '''

            vec = np.dot(M_, np.array([[col_, row_, 1]]).T)
            c = vec[0, 0]
            r = vec[1, 0]
            c_left = int(c)
            c_right = min(int(c + 1), w - 1)
            r_top = int(r)
            r_bottom = min(int(r + 1), h - 1)
            s = c - c_left
            t = r - r_top

            intensity = (1 - s) * (1 - t) * src[r_top, c_left] \
                            + s * (1 - t) * src[r_top, c_right] \
                            + (1 - s) * t * src[r_bottom, c_left] \
                            + s * t * src[r_bottom, c_right]

            dst_back_bilinear[row_, col_] = intensity

    dst_back_bilinear = dst_back_bilinear.astype(np.uint8)

    return dst_back_bilinear, M




def main():
    src = cv2.imread('Lena.png')
    img = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5)

    img_point = img.copy()
    img_point[160,160, :] = [0, 0, 255]

    cv2.imshow('img 201702081', img)
    dst, M = scaling_test(img)

    '''
    #ToDo
    ### 160, 160에 점찍기
    ###row_와 col_을 구하기 위해서 ??? 채우기
    ###딱 한 픽셀에만 점을 찍기 위해 소수의 경우 가장 가까운 위치로 변경 ex : (1.9, 1.8)이 row_와 col_으로 나온 경우 row : 2, col : 2로 변경
    '''
    vec = np.dot(M, np.array([[160, 160, 1]]).T)
    col_ = int(np.round(vec[0,0]))
    row_ = int(np.round(vec[1,0]))
    dst[row_, col_] = [0, 0, 255]

    dst_FM, M_FM = feature_matching(img, src)

    '''
    #ToDo
    ### 160, 160에 점찍기
    ###row_와 col_을 구하기 위해서 ??? 채우기
    ###딱 한 픽셀에만 점을 찍기 위해 소수의 경우 가장 가까운 위치로 변경 ex : (1.9, 1.8)이 row_와 col_으로 나온 경우 row : 2, col : 2로 변경
    '''
    vec = np.dot(M_FM, np.array([[160, 160, 1]]).T)
    col_ = int(np.round(vec[0,0]))
    row_ = int(np.round(vec[1,0]))
    dst_FM[row_, col_] = [0, 0, 255]
    print('No RANSAC distance')
    print('point : ', row_, col_)
    print(L2_distance(np.array([320,320]), np.array([row_,col_])))

    dst_FM_RANSAC, M_FM_RANSAC = feature_matching(img, src, RANSAC=True, threshold_distance=5)

    '''
    #ToDo
    ### 160, 160에 점찍기
    ###row_와 col_을 구하기 위해서 ??? 채우기
    ###딱 한 픽셀에만 점을 찍기 위해 소수의 경우 가장 가까운 위치로 변경 ex : (1.9, 1.8)이 row_와 col_으로 나온 경우 row : 2, col : 2로 변경
    '''
    vec = np.dot(M_FM_RANSAC, np.array([[160, 160, 1]]).T)
    col_ = int(np.round(vec[0,0]))
    row_ = int(np.round(vec[1,0]))
    dst_FM_RANSAC[row_, col_] = [0, 0, 255]

    print('Use RANSAC distance')
    print('point : ', row_, col_)
    print(L2_distance(np.array([320,320]), np.array([row_,col_])))

    print('No RANSAC M')
    print(M_FM)
    print('RANSAC M')
    print(M_FM_RANSAC)

    cv2.imshow('img_point 201702081', img_point)
    cv2.imshow('dst 201702081', dst)
    cv2.imshow('dst_FM 201702081', dst_FM)
    cv2.imshow('dst_FM_RANSAC 201702081', dst_FM_RANSAC)

    cv2.waitKey()


if __name__ == '__main__' :
    main()