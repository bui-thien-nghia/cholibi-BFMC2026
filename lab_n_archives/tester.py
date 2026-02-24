import cv2
import numpy as np
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
from collections import defaultdict

model_src = r'C:\Users\Bui Thien Nghia\Documents\PERSONAL FILES 2\Cholibi-BFMC2026\lab_n_archives\model\obj_tuned_320\weights\best.pt'
video_path = r'C:\Users\Bui Thien Nghia\Documents\PERSONAL FILES 2\Cholibi-BFMC2026\lab_n_archives\Records\bfmc2020_online_3.avi'
model = YOLO(model_src)
print('=====INITIATION COMPLETE=====')

# ================================================== PREDICTING ==================================================
model.track(
    source=video_path,
    # save=True,
    show=True,
    half=True,
    imgsz=416,
    conf=0.7,
    vid_stride=1
)

# =================================================== TRACKING ===================================================
# window_name = 'YOLO26 Tracking'
# # cap = cap_from_youtube(video_path, '240p')
# cap = cv2.VideoCapture(video_path)
# track_history = defaultdict(lambda: [])
# iters = 1

# print('Press \'Q\' key to quit')
# while cap.isOpened():
#     success, frame = cap.read()
#     if success and iters % 1 == 0:
#         result = model.track(
#             source=frame,
#             show=False,
#             half=True,
#             imgsz=320,
#             conf=0.2,
#             persist=True,
#             verbose=False,
#             device=0
#         )[0]

#         if result.boxes and result.boxes.is_track:
#             boxes = result.boxes.xywh.cpu()
#             track_ids = result.boxes.id.int().cpu().tolist()
#             frame = result.plot()

#             for box, track_id in zip(boxes, track_ids):
#                 x, y, w, h = box
#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))
#                 if len(track) > 20:  # retain 30 tracks for 30 frames
#                     track.pop(0)

#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
#         cv2.imshow(window_name, frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     iters += 1

# cap.release()
# cv2.destroyAllWindows()