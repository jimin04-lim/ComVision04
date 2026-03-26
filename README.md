# ComVision 04주차 실습
# OpenCV 실습 과제

## 0401. SIFT를 이용한 특징점 검출 및 시각화
- **설명**: 주어진 이미지(mot_color70.jpg)를 이용하여 SIFT알고리즘을 사용한 특징점 검출
- **요구사항**:
  - cv.SIFT_create()    :SIFT 객체 생성
    - _매개변수를 변경하여 특징점 검출 결과를 비교(특징점이 너무 많다면 nfeatures값 조정)_
  - detectAndCompute()  :특징점 검출
  - cv.drawKeypoints()  :특징점을 이미지에 시각화
    - _flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하면 특징점의 방향과 크기도 표시_
  - matplotib           :원본 이미지와 특징점이 시각화된 이미지를 나란히 출력
- **코드**
  ```python
  ```
- **주요코드**
  ```python
  cv.SIFT_create(nfeatures=500)      #이미지에서 추출할 특징점의 최대 개수를 설정하여 SIFT 알고리즘 객체를 생성
  sift.detectAndCompute(gray, None)  #이미지의 밝기 변화를 분석해 특징점 위치와 해당 점의 특징(기술자)을 동시에 계산
  flags=cv.DRAW_RICH_KEYPOINTS       #단순한 점이 아니라 특징점의 크기(Scale)와 방향(Orientation)을 원과 선으로 상세히 표시
  ```
- **결과물**:



## 0402. SIFT를 이용한 두 영상 간 특징점 매칭
- **설명**: 두 개의 이미지(mot_color70.jpg, mot_color83.jpg)를 입력받아 SIFT 특징점 기반으로 매칭을 수행하고 결과를 시각화
- **요구사항**:
  - cv.imread()                             :두 개의 이미지를 불러옴
  - cv.SIFT_create()                        :특징점 추출
  - cv.BFMatcher()/cv.FlannBasedMatcher()   :두 영상 간 특징점 매칭
    - _FLANN 기반 매칭을 원한다면 FlannBasedMatcher()를 사용_
    - _BFMatcher(cv.NORM_L2, crossCheck=True)을 사용하면 간단한 매칭 가능_
    - _KnnMatch()와 DMatch() 객체를 활용하여 최근접 이웃 거리 비율을 적용하면 매칭 정확도를 높일 수 있음_
  - cv.drawMatches()                        :매칭결과 시각화
  - matplotib                               :매칭 결과 출력
- **코드**
  ```python
  ```
- **주요코드**
  ```python
  cv.BFMatcher(cv.NORM_L2, crossCheck=True)  #모든 특징점을 일일이 비교(Brute-Force)하며, 양방향에서 서로 가장 일치하는 점만 남기도록 설정
  bf.match(des1, des2)                       #두 이미지의 기술자들을 비교하여 가장 거리가 가까운(닮은) 쌍을 찾아냄
  cv.drawMatches(...)                        #두 영상을 가로로 붙이고 서로 대응되는 특징점들을 선으로 연결하여 시각화
  ```
- **결과물**: 


## 0403. 호모그래피를 이용한 이미지 정합
- **설명**: SIFT 특징점을 사용한 두 이미지(img1.jpg, img2.jpg, img3.jpg 중 택 2) 간 대응점 검출 후 호모그래피를 계산하여 하나의 이미지 위에 정렬
- **요구사항**:
  - cv.imread()                :두 개의 이미지를 불러옴
  - cv.SIFT_create()           :특징점 검출
  - cv.BFMatcher()과 knnMatch():특징점 매칭, 좋은 매칭점만 선별
    - _knnMatch()로 두개의 최근접 이웃을 구한 뒤, 거리 비율이 임계값(예: 0.7)미만인 매칭점만 선별_
  - cv.findHomography()        :호모그래피 행렬을 계산
    - _cv.RANSAC을 사용하면 Outlier 영향을 줄일 수 있음_
  - cv.warpPerspective()       :한 이미지를 변환하여 다른 이미지와 정렬
    - _출력 크기는 두 이미지를 합친 파노라마 크기(w1+w2, max(h1,h2))로 설정_
  - 변환 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
- **코드**
  ```python
  ```
- **주요코드**
  ```python
  bf.knnMatch(..., k=2)              #각 특징점마다 가장 닮은 2개를 찾아 거리 비율을 비교함으로써 모호한 매칭을 제거
  cv.findHomography(..., cv.RANSAC)  #잘못 매칭된 점(Outlier)들을 무시하고 다수의 정상 매칭점들을 가장 잘 설명하는 3x3 변환 행렬을 구함
  cv.warpPerspective(...)            #계산된 호모그래피 행렬을 바탕으로 이미지의 시점을 비틀어 다른 이미지와 좌표계를 일치시킴
  ```
- **결과물**:


