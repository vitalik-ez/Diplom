import random
import mediapipe as mp
import cv2
import torch


mpDraw = mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic()
mp_drawing_styles = mp.solutions.drawing_styles

class Model:

    def __init__(self) -> None:
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def posenet_detect(self, image):
        
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        
        metadata = []
        
        if results.pose_landmarks:
            mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            res = results.pose_landmarks.landmark
            for i in range(len(results.pose_landmarks.landmark)):
                x = res[i].x
                y = res[i].y
                z = res[i].z
                tmp = {"x": x, "y": y, "z": z}
                metadata.append(tmp)
            
        return metadata, image


    def holistic_detect(self, image):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(imgRGB)
        
        metadata = []
        
        if results.pose_landmarks and results.face_landmarks:
            #mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            
            mpDraw.draw_landmarks(
                image,
                results.face_landmarks,
                mpHolistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mpDraw.draw_landmarks(
                image,
                results.pose_landmarks,
                mpHolistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())


            res = results.pose_landmarks.landmark
            for i in range(len(results.pose_landmarks.landmark)):
                x = res[i].x
                y = res[i].y
                z = res[i].z
                tmp = {"x": x, "y": y, "z": z}
                metadata.append(tmp)
            
        return metadata, image



    def yolov5_detect(self, image):
        results = self.yolo_model(image)
        metadata = []
        labels = [ i for i in results.pandas().xyxy[0]['name'] ]
        results = results.xyxy[0].cpu().detach().numpy()
        for boundRect, label in zip(results, labels):
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            image = cv2.rectangle(image, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[2]), int(boundRect[3])), color, 2)
            cv2.putText(image, label, (int(boundRect[0]), int(boundRect[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return labels, image
