'''
YoloZone Pose Module
Developed by:
    - Nushan Kodikara
Contact:
    - nushankodi@gmail.com
'''
import numpy as np
import cv2
import ultralytics

class PoseDetector:
    def __init__(self, model="yolov8s-pose.pt"):
        self.model = model
        self.pose = ultralytics.YOLO(self.model)
    
    def find_keypoints(self, img, device="cpu"):
        '''Find the keypoints in an image'''
        self.results = self.pose(img, device=device)
        return self.results[0].keypoints.cpu().numpy()

    def angle_between_3_points(self, keypoints, point1, point2, point3, subject_index=0):
        '''Calculate the angle between three points'''
        pointA = np.array(keypoints.xy[subject_index][point1])
        pointB = np.array(keypoints.xy[subject_index][point2])
        pointC = np.array(keypoints.xy[subject_index][point3])

        radians = np.arctan2(pointC[1] - pointB[1], pointC[0] - pointB[0]) - np.arctan2(pointA[1] - pointB[1], pointA[0] - pointB[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        # Prepare text output directly here
        text = f"{angle:.2f} deg"
        text_position = (int(keypoints.xy[subject_index][point2][0]), int(keypoints.xy[subject_index][point2][1]))

        return angle, text, text_position

    def distance_between_2_points(self, keypoints, point1, point2, subject_index=0):
        '''Calculate the distance between two points'''
        pointA = np.array(keypoints.xy[subject_index][point1])
        pointB = np.array(keypoints.xy[subject_index][point2])

        text = f"{np.linalg.norm(pointA - pointB):.2f}"
        text_position = np.mean([pointA, pointB], axis=0).astype(int)
        pointOutA = tuple(int(x) for x in pointA)
        pointOutB = tuple(int(x) for x in pointB)
        return np.linalg.norm(pointA - pointB), text, text_position, pointOutA, pointOutB

    def angle_between_2_lines(self, keypoints, point1, point2, point3, point4, subject_index=0):
        '''Calculate the angle between two lines'''
        line1 = np.array(keypoints.xy[subject_index][point1]) - np.array(keypoints.xy[subject_index][point2])
        line2 = np.array(keypoints.xy[subject_index][point3]) - np.array(keypoints.xy[subject_index][point4])

        angle = np.arccos(np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2)))
        degrees = angle * 180.0 / np.pi

        text = f"{degrees:.2f} deg"
        # text position is the midpoint of the four points
        text_position = tuple(int(x) for x in np.mean([np.array(keypoints.xy[subject_index][point1]), np.array(keypoints.xy[subject_index][point2]), np.array(keypoints.xy[subject_index][point3]), np.array(keypoints.xy[subject_index][point4])], axis=0).astype(int))
        
        return degrees, text, text_position
    
    def draw_pose_line_check(self, keypoints, point1, point2, subject_index=0):
        '''Check if the line between two points should be drawn'''
        if keypoints.xy[subject_index][point1][0] != 0 and keypoints.xy[subject_index][point1][1] != 0 and keypoints.xy[subject_index][point2][0] != 0 and keypoints.xy[subject_index][point2][1] != 0:
            return True
        return False
    
    def draw_pose(self, keypoints, subject_index=0):
        '''Draw the pose on an image'''
        circle_array = np.array([tuple(int(x) for x in keypoints.xy[subject_index][i]) for i in range(0, keypoints.xy[subject_index].shape[0])])
        
        line_array = []

        # Nose to left eye 0 - 1
        if self.draw_pose_line_check(keypoints, 0, 1, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][0][0]), int(keypoints.xy[subject_index][0][1])), (int(keypoints.xy[subject_index][1][0]), int(keypoints.xy[subject_index][1][1]))))
        # Nose to right eye 0 - 2
        if self.draw_pose_line_check(keypoints, 0, 2, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][0][0]), int(keypoints.xy[subject_index][0][1])), (int(keypoints.xy[subject_index][2][0]), int(keypoints.xy[subject_index][2][1]))))
        # Left eye to left ear 1 - 3
        if self.draw_pose_line_check(keypoints, 1, 3, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][1][0]), int(keypoints.xy[subject_index][1][1])), (int(keypoints.xy[subject_index][3][0]), int(keypoints.xy[subject_index][3][1]))))
        # Right eye to right ear 2 - 4
        if self.draw_pose_line_check(keypoints, 2, 4, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][2][0]), int(keypoints.xy[subject_index][2][1])), (int(keypoints.xy[subject_index][4][0]), int(keypoints.xy[subject_index][4][1]))))

        # Left ear to left shoulder 3 - 5
        if self.draw_pose_line_check(keypoints, 3, 5, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][3][0]), int(keypoints.xy[subject_index][3][1])), (int(keypoints.xy[subject_index][5][0]), int(keypoints.xy[subject_index][5][1]))))
        # Right ear to right shoulder 4 - 6
        if self.draw_pose_line_check(keypoints, 4, 6, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][4][0]), int(keypoints.xy[subject_index][4][1])), (int(keypoints.xy[subject_index][6][0]), int(keypoints.xy[subject_index][6][1]))))

        # Left shoulder to right shoulder 5 - 6
        if self.draw_pose_line_check(keypoints, 5, 6, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][5][0]), int(keypoints.xy[subject_index][5][1])), (int(keypoints.xy[subject_index][6][0]), int(keypoints.xy[subject_index][6][1]))))
        # Left shoulder to left elbow 5 - 7
        if self.draw_pose_line_check(keypoints, 5, 7, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][5][0]), int(keypoints.xy[subject_index][5][1])), (int(keypoints.xy[subject_index][7][0]), int(keypoints.xy[subject_index][7][1]))))
        # Right shoulder to right elbow 6 - 8
        if self.draw_pose_line_check(keypoints, 6, 8, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][6][0]), int(keypoints.xy[subject_index][6][1])), (int(keypoints.xy[subject_index][8][0]), int(keypoints.xy[subject_index][8][1]))))
        # Left elbow to left wrist 7 - 9
        if self.draw_pose_line_check(keypoints, 7, 9, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][7][0]), int(keypoints.xy[subject_index][7][1])), (int(keypoints.xy[subject_index][9][0]), int(keypoints.xy[subject_index][9][1]))))
        # Right elbow to right wrist 8 - 10
        if self.draw_pose_line_check(keypoints, 8, 10, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][8][0]), int(keypoints.xy[subject_index][8][1])), (int(keypoints.xy[subject_index][10][0]), int(keypoints.xy[subject_index][10][1]))))

        # Left shoulder to left hip 5 - 11
        if self.draw_pose_line_check(keypoints, 5, 11, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][5][0]), int(keypoints.xy[subject_index][5][1])), (int(keypoints.xy[subject_index][11][0]), int(keypoints.xy[subject_index][11][1]))))
        # Right shoulder to right hip 6 - 12
        if self.draw_pose_line_check(keypoints, 6, 12, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][6][0]), int(keypoints.xy[subject_index][6][1])), (int(keypoints.xy[subject_index][12][0]), int(keypoints.xy[subject_index][12][1]))))

        # Left hip to right hip 11 - 12
        if self.draw_pose_line_check(keypoints, 11, 12, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][11][0]), int(keypoints.xy[subject_index][11][1])), (int(keypoints.xy[subject_index][12][0]), int(keypoints.xy[subject_index][12][1]))))

        # Left hip to left knee 11 - 13
        if self.draw_pose_line_check(keypoints, 11, 13, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][11][0]), int(keypoints.xy[subject_index][11][1])), (int(keypoints.xy[subject_index][13][0]), int(keypoints.xy[subject_index][13][1]))))
        # Right hip to right knee 12 - 14
        if self.draw_pose_line_check(keypoints, 12, 14, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][12][0]), int(keypoints.xy[subject_index][12][1])), (int(keypoints.xy[subject_index][14][0]), int(keypoints.xy[subject_index][14][1]))))
        # Left knee to left ankle 13 - 15
        if self.draw_pose_line_check(keypoints, 13, 15, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][13][0]), int(keypoints.xy[subject_index][13][1])), (int(keypoints.xy[subject_index][15][0]), int(keypoints.xy[subject_index][15][1]))))
        # Right knee to right ankle 14 - 16
        if self.draw_pose_line_check(keypoints, 14, 16, subject_index):
            line_array.append(((int(keypoints.xy[subject_index][14][0]), int(keypoints.xy[subject_index][14][1])), (int(keypoints.xy[subject_index][16][0]), int(keypoints.xy[subject_index][16][1]))))

        return circle_array, line_array, 


