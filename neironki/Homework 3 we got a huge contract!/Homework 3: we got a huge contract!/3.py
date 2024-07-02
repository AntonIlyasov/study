import os
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

# Пути к директориям
train_directory = 'input/lab3/train'
template_directory = 'input/lab3'
output_directory = 'lab3-result'

# Очистка или создание директории результатов
if os.path.exists(output_directory):
    for root, dirs, files in os.walk(output_directory):
        for file in files:
            os.remove(os.path.join(root, file))
else:
    os.makedirs(output_directory)

# Функция для кластеризации совпадений
def cluster_matches(matches, kp1, kp2):
    if not matches:
        return []
    points = np.array([kp1[m.queryIdx].pt + kp2[m.trainIdx].pt for m in matches])
    kmeans = KMeans(n_clusters=min(2, len(matches) // 5 + 1), random_state=0).fit(points)
    clusters = [[] for _ in range(kmeans.n_clusters)]
    for match, label in zip(matches, kmeans.labels_):
        clusters[label].append(match)
    return clusters

# Обработка и сохранение результатов
def process_and_save(detector, matcher, use_knn=False, ratio_test=False):
    angles = [0, 90, 180, 270]  # Повороты для многопозиционного поиска
    for template_file in os.listdir(template_directory):
        template_img = cv.imread(os.path.join(template_directory, template_file))
        for angle in angles:
            M = cv.getRotationMatrix2D((template_img.shape[1] / 2, template_img.shape[0] / 2), angle, 1)
            rotated_template = cv.warpAffine(template_img, M, (template_img.shape[1], template_img.shape[0]))
            template_gray = cv.cvtColor(rotated_template, cv.COLOR_BGR2GRAY)
            kp1, des1 = detector.detectAndCompute(template_gray, None)

            for train_file in os.listdir(train_directory):
                train_img = cv.imread(os.path.join(train_directory, train_file))
                train_gray = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)
                kp2, des2 = detector.detectAndCompute(train_gray, None)

                if use_knn:
                    matches = matcher.knnMatch(des1, des2, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                else:
                    matches = matcher.match(des1, des2)
                    good_matches = sorted(matches, key=lambda x: x.distance)

                if len(good_matches) > 10:
                    clusters = cluster_matches(good_matches, kp1, kp2)
                    img2_with_boxes = train_img.copy()
                    for cluster in clusters:
                        if len(cluster) >= 4:
                            src_pts = np.float32([kp1[m.queryIdx].pt for m in cluster]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m in cluster]).reshape(-1, 1, 2)
                            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                            if M is not None:
                                h, w = rotated_template.shape[:2]
                                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                                dst = cv.perspectiveTransform(pts, M)
                                img2_with_boxes = cv.polylines(img2_with_boxes, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)

                    output_file = os.path.join(output_directory, f"{os.path.splitext(train_file)[0]}_{os.path.splitext(template_file)[0]}_angle_{angle}.png")
                    cv.imwrite(output_file, img2_with_boxes)

# Инициализация SIFT и BFMatcher с Ratio Test
sift = cv.SIFT_create(nfeatures=1000)  # Увеличение количества точек
bf_sift = cv.BFMatcher()

# Обработка изображений и сохранение результатов
process_and_save(sift, bf_sift, use_knn=True, ratio_test=True)
