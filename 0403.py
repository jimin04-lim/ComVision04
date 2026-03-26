import cv2 as cv # OpenCV 라이브러리 임포트
import numpy as np # 행렬 및 배열 연산을 위한 numpy 임포트
import matplotlib.pyplot as plt # 결과 시각화를 위한 matplotlib 임포트

# 1. 두 개의 이미지 불러오기
img1 = cv.imread('img1.jpg') # 기준이 되는 영상 (왼쪽)
img2 = cv.imread('img3.jpg') # 호모그래피 변환을 적용할 영상 (오른쪽)

# 2. SIFT 특징점 검출 및 기술자 계산
sift = cv.SIFT_create() # SIFT 객체 생성
kp1, des1 = sift.detectAndCompute(img1, None) # img1의 특징점과 기술자 추출
kp2, des2 = sift.detectAndCompute(img2, None) # img2의 특징점과 기술자 추출

# 3. BFMatcher와 knnMatch를 이용한 특징점 매칭
bf = cv.BFMatcher() # Brute-Force 매처 객체 생성
knn_matches = bf.knnMatch(des1, des2, k=2) # 가장 유사한 2개의 대응점 탐색

# 4. Lowe's Ratio Test를 통한 좋은 매칭점 선별 (임계값 0.7)
good_matches = [] # 우수한 매칭점을 담을 리스트
for m, n in knn_matches: # 검색된 두 개의 대응점 후보 중
    if m.distance < 0.7 * n.distance: # 첫 번째 후보가 두 번째보다 충분히 가까우면
        good_matches.append(m) # 좋은 매칭점으로 판단하여 추가

# 5. 호모그래피 행렬 계산 (RANSAC 활용)
if len(good_matches) > 4: # 최소 4개 이상의 대응점이 필요
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # 기준 이미지 좌표
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # 대상 이미지 좌표
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0) # 이상치를 제거하며 변환 행렬 계산

# 6. 이미지 원근 변환 및 정합 (Warping)
h1, w1 = img1.shape[:2] # img1의 높이와 너비
h2, w2 = img2.shape[:2] # img2의 높이와 너비
# 변환 후 출력 크기 설정 (파노라마 형태: 너비 합산, 높이는 최대치)
warped_img = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2))) # img2를 img1 좌표계로 변환
warped_img[0:h1, 0:w1] = img1 # 변환된 이미지의 왼쪽 영역에 원본 img1 삽입

# 7. 매칭 결과 시각화 생성
matching_res = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 8. 최종 결과 출력 (변환 이미지와 매칭 결과를 나란히)
plt.figure(figsize=(20, 10)) # 출력 창 크기 설정
plt.subplot(2, 1, 1), plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)), plt.title('Warped Image (Panorama)') # 정합 결과
plt.subplot(2, 1, 2), plt.imshow(cv.cvtColor(matching_res, cv.COLOR_BGR2RGB)), plt.title('Matching Result') # 매칭 쌍 시각화
plt.tight_layout() # 간격 조절
plt.show() # 화면 표시