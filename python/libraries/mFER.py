import cv2
import mediapipe as mp
import numpy as np

mp_drawing = None
mp_drawing_styles = None
mp_face_mesh = None
cap = None
camera_id = 0
width = 240
height = 320
detection_confidence = 0.5
tracking_confidence = 0.5
fullscreen = True
direction_mode = 0
screen_width = 240
screen_height = 320
show_face_mesh = False
face_mesh_instance = None

# 全域變數來儲存目前情緒狀態 / Global variables to store current emotion state
current_emotion = "No Face"
current_confidence = 0.0
emotion_counters = {
    "Happy": 0,
    "Sad": 0,
    "Angry": 0,
    "Surprised": 0,
    "Neutral": 0,
    "No Face": 0,
    "Detecting...": 0
}

def detect_emotion(landmarks):
    global current_emotion, current_confidence
    
    if landmarks is None:
        current_emotion = "No Face"
        current_confidence = 0.0
        return current_emotion, current_confidence
    
    h, w = landmarks.shape[0], landmarks.shape[1] if len(landmarks.shape) > 1 else 1
    
    left_mouth = landmarks[61] if len(landmarks) > 61 else None
    right_mouth = landmarks[291] if len(landmarks) > 291 else None
    upper_lip = landmarks[13] if len(landmarks) > 13 else None
    lower_lip = landmarks[14] if len(landmarks) > 14 else None
    
    left_eyebrow_inner = landmarks[70] if len(landmarks) > 70 else None
    right_eyebrow_inner = landmarks[300] if len(landmarks) > 300 else None
    left_eyebrow_outer = landmarks[46] if len(landmarks) > 46 else None
    right_eyebrow_outer = landmarks[276] if len(landmarks) > 276 else None
    
    left_eye_top = landmarks[159] if len(landmarks) > 159 else None
    left_eye_bottom = landmarks[145] if len(landmarks) > 145 else None
    right_eye_top = landmarks[386] if len(landmarks) > 386 else None
    right_eye_bottom = landmarks[374] if len(landmarks) > 374 else None
    left_eye = landmarks[33] if len(landmarks) > 33 else None
    right_eye = landmarks[362] if len(landmarks) > 362 else None
    
    if all(point is not None for point in [left_mouth, right_mouth, upper_lip, lower_lip, 
                                            left_eye_top, left_eye_bottom, right_eye_top, right_eye_bottom,
                                            left_eyebrow_inner, right_eyebrow_inner]):
        
        mouth_width = abs(right_mouth[0] - left_mouth[0])
        mouth_height = abs(upper_lip[1] - lower_lip[1])
        mouth_curve = (left_mouth[1] + right_mouth[1]) / 2 - upper_lip[1]
        
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        if left_eyebrow_outer is not None and right_eyebrow_outer is not None and left_eye is not None and right_eye is not None:
            left_eyebrow_eye_distance = left_eye[1] - left_eyebrow_outer[1]
            right_eyebrow_eye_distance = right_eye[1] - right_eyebrow_outer[1]
            avg_eyebrow_eye_distance = (left_eyebrow_eye_distance + right_eyebrow_eye_distance) / 2
        else:
            avg_eyebrow_eye_distance = 0.02
        
        if left_eyebrow_inner is not None and right_eyebrow_inner is not None:
            eyebrow_inner_distance = abs(right_eyebrow_inner[0] - left_eyebrow_inner[0])
            left_inner_eye_distance = left_eye[1] - left_eyebrow_inner[1] if left_eye is not None else 0.02
            right_inner_eye_distance = right_eye[1] - right_eyebrow_inner[1] if right_eye is not None else 0.02
            avg_inner_eyebrow_distance = (left_inner_eye_distance + right_inner_eye_distance) / 2
        else:
            eyebrow_inner_distance = 0.1
            avg_inner_eyebrow_distance = 0.02
        
        angry_score = 0
        
        if eyebrow_inner_distance < 0.10:
            angry_score += 2
            
        if avg_inner_eyebrow_distance < 0.025:
            angry_score += 2
            
        if avg_eye_height < 0.020:
            angry_score += 1.5
            
        if mouth_height < 0.010:
            angry_score += 1
            
        if 0.001 < mouth_curve < 0.005:
            angry_score += 1
        
        if avg_eyebrow_eye_distance < 0.018:
            angry_score += 1
        
        if angry_score >= 3:
            confidence = min(0.95, 0.6 + (angry_score * 0.08))
            current_emotion = "Angry"
            current_confidence = confidence
            return current_emotion, current_confidence
        elif mouth_curve < -0.005:
            current_emotion = "Happy"
            current_confidence = 0.85
            return current_emotion, current_confidence
        elif mouth_curve > 0.008:
            current_emotion = "Sad"
            current_confidence = 0.80
            return current_emotion, current_confidence
        elif (mouth_height > mouth_width * 0.25 and mouth_height > 0.015 and 
              avg_eye_height > 0.015 and avg_eyebrow_eye_distance > 0.025):
            current_emotion = "Surprised"
            current_confidence = 0.75
            return current_emotion, current_confidence
        else:
            current_emotion = "Neutral"
            current_confidence = 0.60
            return current_emotion, current_confidence
    
    current_emotion = "Detecting..."
    current_confidence = 0.0
    return current_emotion, current_confidence

def setup_camera(cam_id, w, h):
    global camera_id, width, height, cap
    camera_id = cam_id
    width = w
    height = h
    
    cap = cv2.VideoCapture(camera_id)
    
    if direction_mode == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def set_detection_params(det_conf, track_conf):
    global detection_confidence, tracking_confidence
    detection_confidence = det_conf
    tracking_confidence = track_conf

def set_ui_options(fs, direction, w, h):
    global fullscreen, direction_mode, screen_width, screen_height
    fullscreen = fs
    direction_mode = direction
    screen_width = w
    screen_height = h
    
    if fullscreen:
        cv2.namedWindow('MediaPipe Face Mesh', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Face Mesh', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)

def start_emotion_detection():
    global face_mesh_instance, show_face_mesh
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=detection_confidence, 
        min_tracking_confidence=tracking_confidence
    ) as face_mesh:
        face_mesh_instance = face_mesh
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                    emotion, confidence = detect_emotion(landmarks_array)
                    
                    if show_face_mesh:
                        mp_drawing.draw_landmarks(
                            image=image, 
                            landmark_list=face_landmarks, 
                            connections=mp_face_mesh.FACEMESH_TESSELATION, 
                            landmark_drawing_spec=None, 
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=image, 
                            landmark_list=face_landmarks, 
                            connections=mp_face_mesh.FACEMESH_CONTOURS, 
                            landmark_drawing_spec=None, 
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=image, 
                            landmark_list=face_landmarks, 
                            connections=mp_face_mesh.FACEMESH_IRISES, 
                            landmark_drawing_spec=None, 
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                        )

            if direction_mode == 0:
                image = cv2.flip(image, 1)
                image = cv2.resize(image, (screen_width, screen_height))
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                        emotion, confidence = detect_emotion(landmarks_array)
                        cv2.putText(image, f"Emotion: {emotion}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(image, f"Rule: {confidence:.2f}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        break
                else:
                    cv2.putText(image, "No Face Detected", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                h, w = image.shape[:2]
                cv2.putText(image, "Press 'a' to exit", (5, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(image, "Press 'b' to toggle mesh", (5, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            else:
                image = cv2.flip(image, 1)
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                image = cv2.resize(image, (screen_width, screen_height))
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                        emotion, confidence = detect_emotion(landmarks_array)
                        cv2.putText(image, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(image, f"Rule-based: {confidence:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        break
                else:
                    cv2.putText(image, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                h, w = image.shape[:2]
                cv2.putText(image, "Press 'a' to exit", (w-180, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image, "Press 'b' to toggle mesh", (w-200, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('MediaPipe Face Mesh', image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('a') or key == 27:
                break
            elif key == ord('b'):
                show_face_mesh = not show_face_mesh

def stop_emotion_detection():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def get_current_emotion():
    global current_emotion
    return current_emotion

def get_emotion_confidence():
    global current_confidence
    return current_confidence

def emotion_detected(emotion_name):
    global current_emotion, emotion_counters
    if current_emotion == emotion_name:
        # 增加計數器 / Increment counter
        emotion_counters[emotion_name] = emotion_counters.get(emotion_name, 0) + 1
        return True
    return False

def get_emotion_count(emotion_name):
    global emotion_counters
    return emotion_counters.get(emotion_name, 0)

def reset_emotion_counter():
    global emotion_counters
    emotion_counters = {
        "Happy": 0,
        "Sad": 0,
        "Angry": 0,
        "Surprised": 0,
        "Neutral": 0,
        "No Face": 0,
        "Detecting...": 0
    }
