import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(path):
    lines = []
    f = open(path)
    while True:
        line = list(map(float, f.readline().split()))
        if not line : break
        lines.append(line)
    f.close()
    return np.array(lines)

def load_data(path):
    
    pts2d_a = read_file(path+'pts2d-pic_a.txt')
    pts2d_b = read_file(path+'pts2d-pic_b.txt')
    pts3d = read_file(path+'pts3d.txt')
    
    pic_a = cv2.imread(path+'pic_a.jpg')
    pic_b = cv2.imread(path+'pic_b.jpg')

    pic_1_1 = cv2.imread(path+'test1/1.jpg')
    pic_1_2 = cv2.imread(path+'test1/2.jpg')
    pic_2_1 = cv2.imread(path+'test2/1.jpg')
    pic_2_2 = cv2.imread(path+'test2/2.jpg')
    pic_3_1 = cv2.imread(path+'test3/1.jpg')
    pic_3_2 = cv2.imread(path+'test3/2.jpg')
    pic_4_1 = cv2.imread(path+'test4/1.jpg')
    pic_4_2 = cv2.imread(path+'test4/2.jpg')
    print(pic_2_2)
    
    return pts2d_a, pts2d_b, pts3d, pic_a, pic_b, pic_1_1, pic_1_2, pic_2_1, pic_2_2, pic_3_1, pic_3_2, pic_4_1, pic_4_2

def camera_matrix(pts2d, pts3d):
    row_3d = pts3d.shape[0]
    p_one = np.ones(row_3d).reshape((row_3d, 1))
    pts2d = np.hstack([pts2d, p_one])
    pts3d = np.hstack([pts3d, p_one])

    p_zero = [0 for _ in range(4)]
    mat = []
    for i in range(row_3d):
        mat.append(np.concatenate((pts3d[i], p_zero, (-1 * pts2d[i][0]) * pts3d[i])))
        mat.append(np.concatenate((p_zero, pts3d[i], (-1 * pts2d[i][1]) * pts3d[i])))
    
    U, s, V = np.linalg.svd(mat)
    ans = V[-1].reshape((3, 4))
    return ans

def chaged_3d_to_2d(pts3d, P):
    changed_2d = []
    row_3d = pts3d.shape[0]
    p_one = np.ones(row_3d).reshape((row_3d, 1))
    pts3d = np.hstack([pts3d, p_one])
    for point in pts3d:
        a = np.matmul(P, point)
        changed_2d.append(a/a[2])
    
    return changed_2d

def run_projection(pts2d, pts3d):
    P = camera_matrix(pts2d, pts3d)
    changed_2d = chaged_3d_to_2d(pts3d, P)
    return changed_2d

def fundamental_matrix(from_pts, d_pts):
    mat = []
    for f_pt, d_pt in zip(from_pts, d_pts):
        x_f, y_f = f_pt
        x_d, y_d = d_pt
        mat_a = [x_d * x_f,
                x_d * y_f,
                x_d, 
                y_d * x_f, 
                y_d * y_f, 
                y_d, 
                x_f, 
                y_f, 
                1]
        mat.append(mat_a)
    
    U, s, V = np.linalg.svd(mat)
    F = V[-1].reshape((3,3))
    return F

def make_line(pts, F):
    row_pt = pts.shape[0]
    p_one = np.ones(row_pt).reshape((row_pt, 1))
    pts = np.hstack([pts, p_one])

    lines = []
    for pt in pts:
        mat = np.matmul(F, pt)
        lines.append(mat)
    return lines

def run_epipolar(from_pts, pts, img):
    '''
    from_pts : used pts
    pts : drawed pts
    '''
    F = fundamental_matrix(from_pts, pts)
    line_list = make_line(from_pts, F)

    # draw
    draw_img = img[:]

    for line in line_list:
        x1 = 0
        x2 = draw_img.shape[1]
        y1 = int(-(line[2] / line[1]))
        y2 = int(-(((line[0] / line[1]) * x2) + (line[2] / line[1])))

        color = (255, 0, 0)
        draw_img = cv2.line(draw_img, (x1, y1), (x2, y2), color, 2)

    for point in pts:
        draw_img = cv2.circle(draw_img, (int(point[0]), int(point[1])), 3, (0, 0, 255), 1)

    return draw_img

def ORB(pic_1, pic_2):
    pic_1 = cv2.resize(pic_1, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    pic_2 = cv2.resize(pic_2, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    basic_pic_1, basic_pic_2 = pic_1[:], pic_2[:]
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(pic_1, None)
    kp2, des2 = orb.detectAndCompute(pic_2, None)
    
    pic_1 = cv2.drawKeypoints(pic_1, kp1, None)
    pic_2 = cv2.drawKeypoints(pic_2, kp2, None)
    cv2.imshow('test1', pic_1)
    cv2.imshow('test2', pic_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return kp1, kp2, pic_1, pic_2, basic_pic_1, basic_pic_2, des1, des2

def kp_to_array(kp1, kp2):
    array1 = []
    array2 = []
    for i in kp1:
        array1.append(i.pt)
    for j in kp2:
        array2.append(j.pt)
    return np.array(array1), np.array(array2)

def draw_maching(lines, img):
    draw_img = img[:]

    for line in lines:
        x1 = 0
        x2 = draw_img.shape[1]
        y1 = int(-(line[2] / line[1]))
        y2 = int(-(((line[0] / line[1]) * x2) + (line[2] / line[1])))

        color = (255, 0, 0)
        draw_img = cv2.line(draw_img, (x1, y1), (x2, y2), color, 1)
    cv2.imshow('draw_img', draw_img)
    cv2.waitKey(0)

    return draw_img

def remove_kp(pt_1, img):
    new_pt_1 = []
    for pt in pt_1:
        try:
            if img[int(pt[0])][int(pt[1])][0] == 255 and img[int(pt[0])][int(pt[1])][1] == 0 and img[int(pt[0])][int(pt[1])][2] == 0:
                new_pt_1.append(pt)
        except:
            pass
    return new_pt_1


def two_img_matching(kp1, kp2, removed_pt2, pic_1, pic_2, des1, des2):
    key_list1 = []
    key_list2 = []

    for i, kp in enumerate(kp2):
        for pt in removed_pt2:
            if kp.pt[0] == pt[0] and kp.pt[1] == pt[1]:
                key_list1.append(kp1[i])
                key_list2.append(kp2[i])

    pic_1 = cv2.drawKeypoints(pic_1, key_list1, None)
    pic_2 = cv2.drawKeypoints(pic_2, key_list2, None)
    cv2.imshow('test1', pic_1)
    cv2.imshow('test2', pic_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv2.drawMatches(pic_1, key_list1, pic_2, key_list2, matches[:10],None,flags=2)

    img3 = cv2.drawMatches(pic_1, kp1, pic_2, kp2, matches[:10],None,flags=2)
    
    plt.imshow(img3), plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def matching(pic_1, pic_2):

    kp1, kp2, pic_1, pic_2, basic_pic_1, basic_pic_2, des1, des2 = ORB(pic_1, pic_2)
    pt_1, pt_2 = kp_to_array(kp1, kp2)

    # ret, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC)
    # print(ret, mask)

    F1 = fundamental_matrix(pt_1, pt_2)
    F2 = fundamental_matrix(pt_2, pt_1)
    
    line_list1 = make_line(pt_1, F1)
    line_list2 = make_line(pt_2, F2)
    
    draw_img1 = draw_maching(line_list1, pic_1)
    draw_img2 = draw_maching(line_list2, pic_2)

    removed_pt2 = remove_kp(pt_2, draw_img2)

    two_img_matching(kp1, kp2, removed_pt2, basic_pic_1, basic_pic_2, des1, des2)


    
    



def run():

    pts2d_a, pts2d_b, pts3d, pic_a, pic_b, pic_1_1, pic_1_2, pic_2_1, pic_2_2, pic_3_1, pic_3_2, pic_4_1, pic_4_2 = load_data('./data/')

    list_3d_2da = run_projection(pts2d_a, pts3d)
    list_3d_2db = run_projection(pts2d_b, pts3d)
    # for checking
    for i in list_3d_2da:
        print(i)
    print("@@@@@@@@@@@@@@@@@")
    for i in list_3d_2db:
        print(i)


    # find error
    mean_error_a = 0
    total_point_a = 0
    for i in range(len(list_3d_2da)):
        error_a = np.square(list_3d_2da[i][0] - pts2d_a[i][0]).sum()
        mean_error_a += error_a
        error_a = np.square(list_3d_2da[i][1] - pts2d_a[i][1]).sum()
        mean_error_a += error_a
    total_point_a += len(pts2d_a*2)
    print("error a : {0}".format(mean_error_a/total_point_a))

    mean_error_b = 0
    total_point_b = 0
    for i in range(len(list_3d_2db)):
        error_b = np.square(list_3d_2db[i][0] - pts2d_b[i][0]).sum()
        mean_error_b += error_b
        error_b = np.square(list_3d_2db[i][1] - pts2d_b[i][1]).sum()
        mean_error_b += error_b
    total_point_b += len(pts2d_b*2)
    print("error b : {0}".format(mean_error_b/total_point_b))

    draw_img = run_epipolar(pts2d_a, pts2d_b, pic_b)
    # for checking
    cv2.imshow('draw_img', draw_img)
    cv2.waitKey(0)

    matching(pic_1_1, pic_1_2)
    matching(pic_2_1, pic_2_2)
    matching(pic_3_1, pic_3_2)
    matching(pic_4_1, pic_4_2)
    



    

run()