from systemMode import SystemModeRebuilt

class StateChanger:
    def __init__(self):
        self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL
        self.classes = [
            'pedestrian',
            'cyclist',
            'car',
            'bus',
            'truck',
            'red_light',
            'yellow_light',
            'green_light',
            'crosswalk_sign',
            'enter_highway_sign',
            'leave_highway_sign',
            'oneway_sign',
            'parking_sign',
            'priority_sign',
            'noentry_sign',
            'roundabout_sign',
            'stop_sign'
        ]
        self.idx_to_cls = {
            0: 'pedestrian',
            1: 'cyclist',
            2: 'car',
            3: 'bus',
            4: 'truck',
            5: 'red_light',
            6: 'yellow_light',
            7: 'green_light',
            8: 'crosswalk_sign',
            9: 'enter_highway_sign',
            10: 'leave_highway_sign',
            11: 'oneway_sign',
            12: 'parking_sign',
            13: 'priority_sign',
            14: 'noentry_sign',
            15: 'roundabout_sign',
            16: 'stop_sign'
        }
        # To be tuned
        self.det_threshold = {
            'pedestrian': 7,
            'cyclist': 7,
            'car': 7,
            'bus': 7,
            'truck': 7,
            'red_light': 7,
            'yellow_light': 7,
            'green_light': 7,
            'crosswalk_sign': 7,
            'enter_highway_sign': 7,
            'leave_highway_sign': 7,
            'oneway_sign': 7,
            'parking_sign': 7,
            'priority_sign': 7,
            'noentry_sign': 7,
            'roundabout_sign': 7,
            'stop_sign': 7
        }
        self.cur_dets = {key: 0 for key in [
            'pedestrian',
            'cyclist',
            'car',
            'bus',
            'truck',
            'red_light',
            'yellow_light',
            'green_light',
            'crosswalk_sign',
            'enter_highway_sign',
            'leave_highway_sign',
            'oneway_sign',
            'parking_sign',
            'priority_sign',
            'noentry_sign',
            'roundabout_sign',
            'stop_sign'
        ]}

    def record_detection(self, idxes, boxes):
        '''Handles detection results'''
        def get_max_cnt(cls, coeff=1):
            return int(self.det_threshold[cls] * coeff)
        
        # To be tuned
        def aspect_ratio_met(box, aspect_ratio, err_rate):
            return aspect_ratio * (1 - err_rate) <= box[-2] / box[-1] <= aspect_ratio * (1 + err_rate)

        dets = [self.idx_to_cls[i] for i in idxes]
        accepted_dets = []
        for d, b in zip(dets, boxes):
            if d in self.classes[:5]:
                accepted_dets.append(d)
            elif d in self.classes[5:7] and aspect_ratio_met(b, 1/3.5, 0.4):
                accepted_dets.append(d)
            elif d in self.classes[7:]:
                if d in ['enter_highway_sign', 'leave_highway_sign'] and aspect_ratio_met(b, 2/3, 0.4):
                    accepted_dets.append(d)
                elif aspect_ratio_met(b, 1/1, 0.4):
                    accepted_dets.append(d)
        
        for c in list(self.cur_dets.keys()):
            if c in accepted_dets:
                self.cur_dets[c] += 1 if self.cur_dets[c] < get_max_cnt(c, 2) else 0
            else:
                self.cur_dets[c] -= 1 if self.cur_dets[c] > 0 else 0

    def change_state(self):
        '''Handles changes based on the detection recorder'''
        # threshold check util
        def threshold_met(cls):
            return self.cur_dets[cls] >= self.det_threshold[cls]
        
        # traffic light handling
        if threshold_met('red_light'):
            self.cur_state = SystemModeRebuilt.STOP
        elif threshold_met('yellow_light') and self.cur_state != SystemModeRebuilt.STOP:
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_SLOW
        elif threshold_met('green_light'):
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL

        # traffic sign handling
        elif threshold_met('stop_sign'):
            self.cur_state = SystemModeRebuilt.STOP
        elif threshold_met('noentry_sign'): # add intersection detection here
            self.cur_state = SystemModeRebuilt.STOP
        elif threshold_met('crosswalk_sign'):
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_SLOW
        elif threshold_met('oneway_sign'): # consider adding a strict go forward only OR construct for this case
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL
        elif threshold_met('priority_sign'):
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL
        elif threshold_met('leave_highway_sign'):
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL
        elif threshold_met('roundabout_sign'): # build a turn recognition based on route detection
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL or SystemModeRebuilt.TURN
        elif threshold_met('enter_highway_sign'):
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_FAST
        elif threshold_met('parking_sign'):
            self.cur_state = SystemModeRebuilt.PARKING

        # object handling
        elif threshold_met('pedestrian') or threshold_met('cyclist'):
            self.cur_state = SystemModeRebuilt.STOP
        elif threshold_met('car') or threshold_met('truck') or threshold_met('bus'): # build a decisor based on movement/distance tracking
            self.cur_state = SystemModeRebuilt.OVERTAKING or SystemModeRebuilt.TAILING

        # No detection
        else:
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL

    def _get_state(self):
        '''Returns a string implying the current car's mode'''
        return self.cur_state
    

# EXAMPLE AS BELOW
if __name__ == '__main__':
    import cv2
    import time
    from ultralytics import YOLO

    model = YOLO('path/to/model')
    video_path = r'path/to/video'

    results = model.track(
        source=video_path,
        # show=True,
        half=True,
        imgsz=416,
        conf=0.75,
        vid_stride=1,
        # save=True,
        verbose=False
    )

    max = 0
    res = None
    coordinates = (10, 50) # Bottom-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    outline_color = (255, 255, 255)
    color = (0, 0, 0) # Green color in BGR
    thickness = 2

    mode_changer = StateChanger()
    for i, r in enumerate(results):
        # StateChanger currently adapts only to results provided by YOLO
        mode_changer.record_detection(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())
        mode_changer.change_state()
        cur_state = mode_changer._get_state()

        img = r.plot()
        cv2.putText(img, cur_state.value['mode'], coordinates, font, fontScale, outline_color, thickness + 4, cv2.LINE_AA)
        cv2.putText(img, cur_state.value['mode'], coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('bruh', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if len(r.boxes.cls) > max:
            res = r
            max = len(r.boxes.cls)
        time.sleep(0.03)

    cv2.destroyAllWindows()

    pass