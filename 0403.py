import cv2 as cv # OpenCV 임포트
import numpy as np # 행렬 연산을 위한 numpy 임포트
import matplotlib.pyplot as plt # 시각화 임포트

# 1. 이미지 로드 및 특징점 추출
img1 = cv.imread('img1.jpg') # 기준 이미지
img2 = cv.imread('img3.jpg') # 변환할 이미지
sift = cv.SIFT_create() # SIFT 생성
kp1, des1 = sift.detectAndCompute(img1, None) # 특징 추출
kp2, des2 = sift.detectAndCompute(img2, None) # 특징 추출

# 2. KNN 매칭 및 거리 비율 검사 (Lowe's ratio test)
bf = cv.BFMatcher() # 매처 생성
knn_matches = bf.knnMatch(des1, des2, k=2) # 가장 유사한 2개의 후보를 찾음
good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance] # 임계값 0.7 이내의 우수한 매칭점만 선별

# 3. 호모그래피 계산 (최소 4개 이상의 매칭점 필요)
if len(good_matches) > 4: # 매칭점이 충분할 때만 실행
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # 기준점 좌표
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # 대상점 좌표
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0) # RANSAC 알고리즘으로 호모그래피 행렬 계산

# 4. 이미지 원근 변환 (Warping)
h, w = img1.shape[:2] # 이미지 크기 획득
res = cv.warpPerspective(img2, H, (w * 2, h)) # 호모그래피를 적용해 이미지를 변환하고 결합 공간 확보
res[0:h, 0:w] = img1 # 변환된 이미지 옆에 기준 이미지 배치

# 5. 결과 출력
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB)) # 파노라마 결과 출력
plt.title('Homography Stitched Image') # 제목
plt.show() # 출력