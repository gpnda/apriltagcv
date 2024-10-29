import numpy as np
import cv2
import pupil_apriltags
import glob


image = cv2.imread("./02.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# define the AprilTags detector options and then detect the AprilTags
# in the input image
print("[INFO] detecting AprilTags...")
#options = pupil_apriltags.DetectorOptions(families="tag36h11")
detector = pupil_apriltags.Detector(families = "tag36h11")
results = detector.detect(gray)

print("[INFO] {} total AprilTags detected".format(len(results)))

print(results)

print(results[0].corners[1][0])
print(results[0].corners[1][1])

print(int(results[0].corners[0][0]), int(results[0].corners[0][1]))
print(int(results[0].corners[1][0]), int(results[0].corners[1][1]))






color = (0,255,0)
thickness = 3

point1 = (int(results[0].corners[0][0]), int(results[0].corners[0][1]))
point2 = (int(results[0].corners[1][0]), int(results[0].corners[1][1]))
point3 = (int(results[0].corners[2][0]), int(results[0].corners[2][1]))
point4 = (int(results[0].corners[3][0]), int(results[0].corners[3][1]))

image = cv2.line(image, point1, point2, color, thickness)
image = cv2.line(image, point2, point3, color, thickness)
image = cv2.line(image, point3, point4, color, thickness)
image = cv2.line(image, point4, point1, color, thickness)













object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
], dtype=np.float32)
 

# Создаем объект камеры (по умолчанию)
camera_matrix = np.array([
    [1000, 0, 640 / 2],
    [0, 1000, 480 / 2],
    [0, 0, 1]
], dtype=np.float32)

# Создаем вектор искажений (по умолчанию)
distortion_coefficients = np.zeros((4, 1), dtype=np.float32)



# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv2.solvePnP(object_points, results[0].corners, camera_matrix, distortion_coefficients)

# project 3D points to image plane
# imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs)

# image = draw(image,corners2,imgpts)
point_3d = np.array([0.5, 0.5, 0.5, 1]) # 3D-точка
point_2d, _ = cv2.projectPoints(
    point_3d[None, :3], 
    rvecs, 
    tvecs, 
    camera_matrix, 
    distortion_coefficients
)
point_2d = point_2d[0][0] # Извлекаем координаты 2D-точки

# Вывод результата
print(f"Матрица трансформации:\n{tvecs}")
print(f"Проекция 3D-точки на 2D:\n{point_2d}")


point_center = (int(point_2d[0]), int(point_2d[1]))
image = cv2.line(image, point1, point_center, color, thickness)
image = cv2.line(image, point2, point_center, color, thickness)
image = cv2.line(image, point3, point_center, color, thickness)
image = cv2.line(image, point4, point_center, color, thickness)











cv2.imshow('image',image)
k = cv2.waitKey(0)
















#cv::Point(results[0]["corners"][0][1],0).corners,imgpts)

# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img


# image = cv2.line(imagecv::Point(results[0]["corners"][0][1],0).corners,imgpts)
# cv2.imshow('image',image)






# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)



# for fname in glob.glob('01.png'):
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
 
#     if ret == True:
#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
 
#         # Find the rotation and translation vectors.
#         ret,rvecs, tvecs = cv2.solvePnP(objp, corners2)
 
#         # project 3D points to image plane
#         imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs)
 
#         img = draw(img,corners2,imgpts)
#         cv2.imshow('img',img)
#         k = cv2.waitKey(0) & 0xFF
#         if k == ord('s'):
#             cv2.imwrite(fname[:6]+'.png', img)
 
# cv2.destroyAllWindows()


















# https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
# https://ftc-docs.firstinspires.org/en/latest/programming_resources/index.html#advanced-topics
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e


# pip install opencv-contrib-python

