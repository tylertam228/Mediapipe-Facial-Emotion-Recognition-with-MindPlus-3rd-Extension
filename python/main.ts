//% color="#ffde34" iconWidth=50 iconHeight=40
// MediaPipe 臉部情緒辨識積木庫 v1.0 / MediaPipe Facial Emotion Recognition blocks v1.0
declare const Generator: any;

//% color="#ffde34" iconWidth=50 iconHeight=40
namespace mediapipe_fer {
    
    // Helper function to determine if a value is a string literal
    function isStringLiteral(val: string): boolean {
        return (val.startsWith("'") && val.endsWith("'")) || (val.startsWith('"') && val.endsWith('"'));
    }

    // === 基本設定積木 / Basic Configuration Blocks ===

    //% block="初始化 MediaPipe 臉部情緒辨識" blockType="command"
    //% group="基本設定"
    export function initEmotionDetection(parameter: any, block: any): void {
        Generator.addImport("import mFER");
        Generator.addImport("import cv2");
        Generator.addImport("import mediapipe as mp");
        Generator.addImport("import numpy as np");

        Generator.addCode(`mp_drawing = mp.solutions.drawing_utils`);
        Generator.addCode(`mp_drawing_styles = mp.solutions.drawing_styles`);
        Generator.addCode(`mp_face_mesh = mp.solutions.face_mesh`);
        Generator.addCode(`show_face_mesh = False`);
    }

    //% block="設定攝影機來源 [CAMERA_ID] 解析度 [WIDTH]x[HEIGHT]" blockType="command"
    //% CAMERA_ID.shadow="number" CAMERA_ID.defl=0
    //% WIDTH.shadow="number" WIDTH.defl=240
    //% HEIGHT.shadow="number" HEIGHT.defl=320
    //% group="基本設定"
    export function setupCamera(parameter: any, block: any): void {
        let cameraId = parameter.CAMERA_ID.code;
        let width = parameter.WIDTH.code;
        let height = parameter.HEIGHT.code;
        Generator.addCode(`camera_id = ${cameraId}`);
        Generator.addCode(`width = ${width}`);
        Generator.addCode(`height = ${height}`);
        Generator.addCode(`cap = cv2.VideoCapture(camera_id)`);
        Generator.addCode(`cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)`);
        Generator.addCode(`cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)`);
        Generator.addCode(`cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`);
    }

    //% block="設定偵測參數 偵測信心度 [DETECTION_CONFIDENCE] 追蹤信心度 [TRACKING_CONFIDENCE]" blockType="command"
    //% DETECTION_CONFIDENCE.shadow="number" DETECTION_CONFIDENCE.defl="0.5"
    //% TRACKING_CONFIDENCE.shadow="number" TRACKING_CONFIDENCE.defl="0.5"
    //% group="基本設定"
    export function setDetectionParams(parameter: any, block: any): void {
        let detectionConf = parameter.DETECTION_CONFIDENCE.code;
        let trackingConf = parameter.TRACKING_CONFIDENCE.code;
        Generator.addCode(`detection_confidence = ${detectionConf}`);
        Generator.addCode(`tracking_confidence = ${trackingConf}`);
    }

    //% block="設定 UI 選項 全螢幕 [FULLSCREEN] 方向模式 [DIRECTION] 螢幕寬度 [WIDTH] 高度 [HEIGHT]" blockType="command"
    //% FULLSCREEN.shadow="dropdown" FULLSCREEN.options="BOOLEAN_TYPE"
    //% DIRECTION.shadow="dropdown" DIRECTION.options="DIRECTION"
    //% WIDTH.shadow="number" WIDTH.defl=240
    //% HEIGHT.shadow="number" HEIGHT.defl=320
    //% group="UI 設定"
    export function setUiOptions(parameter: any, block: any): void {
        let fullscreen = parameter.FULLSCREEN.code;
        let direction = parameter.DIRECTION.code;
        let width = parameter.WIDTH.code;
        let height = parameter.HEIGHT.code;
        
        // 轉換為 Python 布林值 / Convert to Python boolean
        fullscreen = fullscreen.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
        
        Generator.addCode(`fullscreen = ${fullscreen}`);
        Generator.addCode(`direction_mode = ${direction}`);
        Generator.addCode(`screen_width = ${width}`);
        Generator.addCode(`screen_height = ${height}`);
        Generator.addCode(``);
        Generator.addCode(`if fullscreen:`);
        Generator.addCode(`    cv2.namedWindow('MediaPipe Face Mesh', cv2.WND_PROP_FULLSCREEN)`);
        Generator.addCode(`    cv2.setWindowProperty('MediaPipe Face Mesh', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)`);
        Generator.addCode(`else:`);
        Generator.addCode(`    cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)`);
    }

    //% block="開始即時情緒偵測" blockType="command"
    //% group="基本設定"
    export function startEmotionDetection(parameter: any, block: any): void {
        Generator.addCode(`with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as face_mesh:`);
        Generator.addCode(`    while cap.isOpened():`);
        Generator.addCode(`        success, image = cap.read()`);
        Generator.addCode(`        if not success:`);
        Generator.addCode(`            continue`);
        Generator.addCode(``);
        Generator.addCode(`        image.flags.writeable = False`);
        Generator.addCode(`        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`);
        Generator.addCode(`        results = face_mesh.process(image)`);
        Generator.addCode(``);
        Generator.addCode(`        image.flags.writeable = True`);
        Generator.addCode(`        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)`);
        Generator.addCode(``);
        Generator.addCode(`        if results.multi_face_landmarks:`);
        Generator.addCode(`            for face_landmarks in results.multi_face_landmarks:`);
        Generator.addCode(`                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])`);
        Generator.addCode(`                emotion, confidence = mFER.detect_emotion(landmarks_array)`);
        Generator.addCode(``);
        Generator.addCode(`                if show_face_mesh:`);
        Generator.addCode(`                    mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())`);
        Generator.addCode(`                    mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())`);
        Generator.addCode(`                    mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())`);
        Generator.addCode(``);
        Generator.addCode(`        if direction_mode == 0:`);
        Generator.addCode(`            image = cv2.flip(image, 1)`);
        Generator.addCode(`            image = cv2.resize(image, (screen_width, screen_height))`);
        Generator.addCode(``);
        Generator.addCode(`            if results.multi_face_landmarks:`);
        Generator.addCode(`                for face_landmarks in results.multi_face_landmarks:`);
        Generator.addCode(`                    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])`);
        Generator.addCode(`                    emotion, confidence = mFER.detect_emotion(landmarks_array)`);
        Generator.addCode(`                    cv2.putText(image, f"Emotion: {emotion}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)`);
        Generator.addCode(`                    cv2.putText(image, f"Rule: {confidence:.2f}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)`);
        Generator.addCode(`                    break`);
        Generator.addCode(`            else:`);
        Generator.addCode(`                cv2.putText(image, "No Face Detected", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)`);
        Generator.addCode(``);
        Generator.addCode(`            h, w = image.shape[:2]`);
        Generator.addCode(`            cv2.putText(image, "Press 'a' to exit", (5, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)`);
        Generator.addCode(`            cv2.putText(image, "Press 'b' to toggle mesh", (5, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)`);
        Generator.addCode(``);
        Generator.addCode(`        else:`);
        Generator.addCode(`            image = cv2.flip(image, 1)`);
        Generator.addCode(`            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)`);
        Generator.addCode(`            image = cv2.resize(image, (screen_width, screen_height))`);
        Generator.addCode(``);
        Generator.addCode(`            if results.multi_face_landmarks:`);
        Generator.addCode(`                for face_landmarks in results.multi_face_landmarks:`);
        Generator.addCode(`                    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])`);
        Generator.addCode(`                    emotion, confidence = mFER.detect_emotion(landmarks_array)`);
        Generator.addCode(`                    cv2.putText(image, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)`);
        Generator.addCode(`                    cv2.putText(image, f"Rule-based: {confidence:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)`);
        Generator.addCode(`                    break`);
        Generator.addCode(`            else:`);
        Generator.addCode(`                cv2.putText(image, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)`);
        Generator.addCode(``);
        Generator.addCode(`            h, w = image.shape[:2]`);
        Generator.addCode(`            cv2.putText(image, "Press 'a' to exit", (w-180, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)`);
        Generator.addCode(`            cv2.putText(image, "Press 'b' to toggle mesh", (w-200, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)`);
        Generator.addCode(``);
        Generator.addCode(`        cv2.imshow('MediaPipe Face Mesh', image)\n`);
    }

    //% block="停止情緒偵測 (點選A鍵關閉，點選B鍵切換顯示面部網格)" blockType="command"
    //% group="基本設定"
    export function stopEmotionDetection(parameter: any, block: any): void {
        Generator.addCode(`\n        key = cv2.waitKey(5) & 0xFF`);
        Generator.addCode(`        if key == ord('a') or key == 27:`);
        Generator.addCode(`            break`);
        Generator.addCode(`        elif key == ord('b'):`);
        Generator.addCode(`            show_face_mesh = not show_face_mesh`);
        Generator.addCode(``);
        Generator.addCode(`cap.release()`);
        Generator.addCode(`cv2.destroyAllWindows()`);
    }

    // === 情緒偵測積木 / Emotion Detection Blocks ===

    //% block="取得目前偵測到的情緒" blockType="reporter"
    //% group="情緒偵測"
    export function getCurrentEmotion(parameter: any, block: any): any {
        Generator.addCode("mFER.get_current_emotion()");
    }

    //% block="偵測到情緒 [EMOTION]" blockType="boolean"
    //% EMOTION.shadow="dropdown" EMOTION.options="EMOTION"
    //% group="情緒偵測"
    export function emotionDetected(parameter: any, block: any): any {
        let emotion = parameter.EMOTION.code;
        if (!emotion || emotion === '') {
            emotion = '"Happy"';
        }
        
        // 處理字符串引號 / Handle string quotes
        if (!isStringLiteral(emotion)) {
            emotion = `"${emotion}"`;
        }
        
        Generator.addCode(`mFER.emotion_detected(${emotion})`);
    }

    //% block="取得情緒 [EMOTION] 出現次數" blockType="reporter"
    //% EMOTION.shadow="dropdown" EMOTION.options="EMOTION"
    //% group="情緒偵測"
    export function getEmotionCount(parameter: any, block: any): any {
        let emotion = parameter.EMOTION.code;
        if (!emotion || emotion === '') {
            emotion = '"Happy"';
        }
        
        // 處理字符串引號 / Handle string quotes
        if (!isStringLiteral(emotion)) {
            emotion = `"${emotion}"`;
        }
        
        Generator.addCode(`mFER.get_emotion_count(${emotion})`);
    }

    //% block="重置情緒計數器" blockType="reporter"
    //% group="情緒偵測"
    export function resetEmotionCounter(parameter: any, block: any): void {
        Generator.addCode("mFER.reset_emotion_counter()");
    }

    
    //% block="縮排 [INDENT_LEVEL] 層 + 編程輸入: [CODE_INPUT]: " blockType="command"
    //% INDENT_LEVEL.shadow="number" INDENT_LEVEL.defl="1"
    //% CODE_INPUT.shadow="normal" CODE_INPUT.defl="（在此輸入對應編程）"
    //% group="編程輸入"
    export function inputCode(parameter: any, block: any) {
        let indentLevel = parameter.INDENT_LEVEL.code || 1;
        let code = parameter.CODE_INPUT.code;
        // 清理字串
        code = code.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
        let spaces = '    '.repeat(indentLevel);
        Generator.addCode(`${spaces}${code}`);
    }

    //% block="縮排 [INDENT_LEVEL] 層 + [CONDITION_TYPE] [CONDITION]:" blockType="command"
    //% INDENT_LEVEL.shadow="number" INDENT_LEVEL.defl="2"
    //% CONDITION_TYPE.shadow="dropdown" CONDITION_TYPE.options="CONDITION_TYPES"
    //% CONDITION.shadow="string" CONDITION.defl=""
    //% group="編程輸入"
    export function addIndentWithCondition(parameter: any, block: any): void {
        let indentLevel = parameter.INDENT_LEVEL.code || 2;
        let conditionType = parameter.CONDITION_TYPE.code;
        let condition = parameter.CONDITION.code;
        
        // 清理字串
        conditionType = conditionType.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
        if (condition) {
            condition = condition.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
        }
        
        // 計算縮排空格數
        let spaces = '    '.repeat(indentLevel);
        
        // 根據條件類型生成不同的語句
        switch(conditionType) {
            case 'if':
                Generator.addCode(`${spaces}if ${condition || 'True'}:`);
                break;
            case 'elif':
                Generator.addCode(`${spaces}elif ${condition || 'True'}:`);
                break;
            case 'else':
                Generator.addCode(`${spaces}else:`);
                break;
            case 'while':
                Generator.addCode(`${spaces}while ${condition || 'True'}:`);
                break;
            case 'while not':
                Generator.addCode(`${spaces}while not ${condition || 'False'}:`);
                break;
            case 'break':
                Generator.addCode(`${spaces}break`);
                break;
            case 'continue':
                Generator.addCode(`${spaces}continue`);
                break;
            case 'pass':
                Generator.addCode(`${spaces}pass`);
                break;
            default:
                Generator.addCode(`${spaces}if ${condition || 'True'}:`);
        }
    }

    //% block="縮排 [INDENT_LEVEL] 層 + print( [CODE_INPUT] ): " blockType="command"
    //% INDENT_LEVEL.shadow="number" INDENT_LEVEL.defl="1"
    //% CODE_INPUT.shadow="normal" CODE_INPUT.defl="（在此輸入對應編程）"
    //% group="編程輸入"
    export function inputCode_Print(parameter: any, block: any) {
        let indentLevel = parameter.INDENT_LEVEL.code || 1;
        let code = parameter.CODE_INPUT.code;
        // 清理字串
        code = code.replace(/^"|"$/g, '').replace(/^'|'$/g, '');
        let spaces = '    '.repeat(indentLevel);
        Generator.addCode(`${spaces}print(${code})`);
    }
}
