# Robot Vision #5

## Due date : 2020-06-12

### 1.Pts3d.txt – The coordinate of 3D points, pts2d-pic_a.txt – The coordinate of corresponding 2D feature points in pic_a.jpg, pts2d-pix_b.txt – The coordinate of corresponding 2D feature points in pic_b.jpg. The order of points corresponds to each other. 

- a point 예측값과 실제값
![](img/1.png)

- a point 예측값과 실제값
![](img/2.png)

- reprojection error
![](img/3.png)

### 2.Implement to compute the fundamental matrix using pts2d-pic_a.txt and pts2d_pix_b.txt. Draw epipolar lines in pic_b from the feature points of pic_a, and epipolar lines in pic_a from the feature points of pic_b. (pic1-epipolar.jpg is an example where epipolar lines in pic_a from the feature points of pic_b)

![](img/5.png)

### 3 Implement to compute fundamental matrices for four testsets, Episcopal Gaudi, Mount Rushmore, Notre Dame, Woodruff Dorm. Using invariant feature point detector and descriptors (e.g., SIFT or SURF) to find correspondences between two images. Then, compute the fundamental matrix for each pair using RANSAC based fundamental matrix estimation method. (You can not only use the SIFT that is implemented by yourself in Homework#4, but also use the library for SIFT or SURF in opencv or vlfeat. )

#### test img 1

- key point
![](img/6.png)
![](img/7.png)

- drow line
![](img/8.png)
![](img/9.png)

- line이랑 겹치는 부분의 key point만 남겨놓고 지우기

![](img/10.png)
![](img/11.png)

- 결과
![](img/12.png)



#### test img 2

- key point
![](img/13.png)
![](img/14.png)

- drow line
![](img/15.png)
![](img/16.png)

- line이랑 겹치는 부분의 key point만 남겨놓고 지우기

![](img/17.png)
![](img/18.png)

- 결과
![](img/21.png)

#### test img 3

- key point
![](img/22.png)
![](img/23.png)

- drow line
![](img/24.png)
![](img/25.png)

- line이랑 겹치는 부분의 key point만 남겨놓고 지우기

![](img/26.png)
![](img/27.png)

- 결과
![](img/28.png)

#### test img 4

- key point
![](img/29.png)
![](img/30.png)

- drow line
![](img/31.png)
![](img/32.png)

- line이랑 겹치는 부분의 key point만 남겨놓고 지우기

![](img/33.png)
![](img/34.png)

- 결과

![](img/36.png)