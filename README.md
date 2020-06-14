# Robot Vision #5

## Due date : 2020-06-12

### 1.Pts3d.txt – The coordinate of 3D points, pts2d-pic_a.txt – The coordinate of corresponding 2D feature points in pic_a.jpg, pts2d-pix_b.txt – The coordinate of corresponding 2D feature points in pic_b.jpg. The order of points corresponds to each other. 

- a point 예측값과 실제값<br/><br/>
![](img/1.png)<br/>

- b point 예측값과 실제값<br/><br/>
![](img/2.png)<br/>

- reprojection error<br/><br/>
![](img/3.png)<br/>
<br/><br/>

### 2.Implement to compute the fundamental matrix using pts2d-pic_a.txt and pts2d_pix_b.txt. Draw epipolar lines in pic_b from the feature points of pic_a, and epipolar lines in pic_a from the feature points of pic_b. (pic1-epipolar.jpg is an example where epipolar lines in pic_a from the feature points of pic_b)
<br/><br/>
![](img/5.png)
<br/><br/>

### 3 Implement to compute fundamental matrices for four testsets, Episcopal Gaudi, Mount Rushmore, Notre Dame, Woodruff Dorm. Using invariant feature point detector and descriptors (e.g., SIFT or SURF) to find correspondences between two images. Then, compute the fundamental matrix for each pair using RANSAC based fundamental matrix estimation method. (You can not only use the SIFT that is implemented by yourself in Homework#4, but also use the library for SIFT or SURF in opencv or vlfeat. )

#### test img 1

- key point<br/><br/>
![](img/6.png)
![](img/7.png)
<br/><br/>
- drow line<br/><br/>
![](img/8.png)
![](img/9.png)
<br/><br/>
- line이랑 겹치는 부분의 key point만 남겨놓고 지우기<br/><br/>

![](img/10.png)
![](img/11.png)
<br/><br/>
- 결과<br/><br/>
![](img/12.png)
<br/><br/>


#### test img 2

- key point<br/><br/>
![](img/13.png)
![](img/14.png)
<br/><br/>
- drow line<br/><br/>
![](img/15.png)
![](img/16.png)
<br/><br/>
- line이랑 겹치는 부분의 key point만 남겨놓고 지우기<br/><br/>

![](img/17.png)
![](img/18.png)
<br/><br/>
- 결과<br/><br/>
![](img/21.png)
<br/><br/>

#### test img 3

- key point<br/><br/>
![](img/22.png)
![](img/23.png)
<br/><br/>
- drow line<br/><br/>
![](img/24.png)
![](img/25.png)
<br/><br/>
- line이랑 겹치는 부분의 key point만 남겨놓고 지우기<br/><br/>

![](img/26.png)
![](img/27.png)
<br/><br/>
- 결과<br/><br/>
![](img/28.png)
<br/><br/>

#### test img 4

- key point<br/><br/>
![](img/29.png)
![](img/30.png)
<br/><br/>
- drow line<br/><br/>
![](img/31.png)
![](img/32.png)
<br/><br/>
- line이랑 겹치는 부분의 key point만 남겨놓고 지우기<br/><br/>

![](img/33.png)
![](img/34.png)
<br/><br/>
- 결과<br/><br/>

![](img/36.png)
