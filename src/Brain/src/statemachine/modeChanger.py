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
            'pedestrian': 3,
            'cyclist': 3,
            'car': 3,
            'bus': 3,
            'truck': 3,
            'red_light': 3,
            'yellow_light': 3,
            'green_light': 3,
            'crosswalk_sign': 3,
            'enter_highway_sign': 3,
            'leave_highway_sign': 3,
            'oneway_sign': 3,
            'parking_sign': 3,
            'priority_sign': 3,
            'noentry_sign': 3,
            'roundabout_sign': 3,
            'stop_sign': 3
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
        self.lookup_nodes = []
        self.lookup_yaw_diffs = []
        self.cooldown = 0

    def record_detection(self, idxes, boxes):
        def get_max_cnt(cls, coeff=1):
            return int(self.det_threshold[cls] * coeff)
        
        # To be tuned
        def significant_sign(box, aspect_ratio, err_rate, area_threshold):
            aspect_ratio_met =  aspect_ratio * (1 - err_rate) <= box[-2] / box[-1] <= aspect_ratio * (1 + err_rate)
            area_met = box[-2] * box[-1] >= area_threshold
            return aspect_ratio_met and area_met

        dets = [self.idx_to_cls[i] for i in idxes]
        accepted_dets = []
        for d, b in zip(dets, boxes):
            if d in self.classes[:5]:
                accepted_dets.append(d)
            elif d in self.classes[5:7] and significant_sign(b, 1/3.5, 0.9, 0.002):
                accepted_dets.append(d)
            elif d in self.classes[7:]:
                if d in ['enter_highway_sign', 'leave_highway_sign'] and significant_sign(b, 2/3, 0.9, 0.002):
                    accepted_dets.append(d)
                elif significant_sign(b, 1/1, 0.9, 0.006):
                    accepted_dets.append(d)
        
        print(f'Accepted Detections: {accepted_dets}')
        for c in list(self.cur_dets.keys()):
            if c in accepted_dets:
                self.cur_dets[c] += 1 if self.cur_dets[c] < get_max_cnt(c, 2) else 0
            else:
                self.cur_dets[c] -= 1 if self.cur_dets[c] > 0 else 0

    def record_lookup(self, node_degrees, yaw_diffs):
        self.lookup_node_degrees = node_degrees
        self.lookup_yaw_diffs = yaw_diffs

    def change_state(self):
        '''Handles changes based on the detection recorder'''
        # threshold check util
        def threshold_met(cls):
            return self.cur_dets[cls] >= self.det_threshold[cls]

        def turn_met(self):
            return max(self.lookup_yaw_diffs) > 30
        
        # if self._get_state() == SystemModeRebuilt.TURN and self.cooldown > 0:
        #     self.cooldown -= 1
        #     return

        # pedestrian handling
        if threshold_met('pedestrian') or threshold_met('cyclist'):
            self.cur_state = SystemModeRebuilt.STOP
        elif threshold_met('car') or threshold_met('truck') or threshold_met('bus'): # build a decisor based on movement/distance tracking
            self.cur_state = SystemModeRebuilt.OVERTAKING or SystemModeRebuilt.TAILING

        # traffic light handling
        elif threshold_met('red_light'):
            self.cur_state = SystemModeRebuilt.STOP
        elif threshold_met('yellow_light') and self._get_state() != SystemModeRebuilt.STOP:
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
        elif threshold_met('roundabout_sign'): # build a turn recognition based on route detection
            self.cur_state = SystemModeRebuilt.TURN if turn_met(self) else SystemModeRebuilt.LANE_KEEPING_NORMAL
        elif threshold_met('enter_highway_sign'):
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_FAST
        elif threshold_met('leave_highway_sign') and self._get_state() == SystemModeRebuilt.LANE_KEEPING_FAST:
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL
        elif threshold_met('parking_sign'):
            self.cur_state = SystemModeRebuilt.PARKING

        # No detection (except for some that needs retaining states)
        elif turn_met(self):
            self.cur_state = SystemModeRebuilt.TURN
            # self.cooldown = 3
        elif self._get_state() not in [SystemModeRebuilt.LANE_KEEPING_FAST, SystemModeRebuilt.LANE_KEEPING_NORMAL,
                                       SystemModeRebuilt.OVERTAKING, SystemModeRebuilt.TAILING, SystemModeRebuilt.PARKING]:
            self.cur_state = SystemModeRebuilt.LANE_KEEPING_NORMAL

    def _get_state(self):
        return self.cur_state
    
    def _get_cooldown(self):
        return self.cooldown

# EXAMPLE AS BELOW
# if __name__ == '__main__':2
    # import cv2
    # import time
    # from ultralytics import YOLO

    # model = YOLO('path/to/model')
    # video_path = r'path/to/video'

    # results = model.track(
    #     source=video_path,
    #     # show=True,
    #     half=True,
    #     imgsz=416,
    #     conf=0.75,
    #     vid_stride=1,
    #     # save=True,
    #     verbose=False
    # )

    # max = 0
    # res = None
    # coordinates = (10, 50) # Bottom-left corner
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 1.5
    # outline_color = (255, 255, 255)
    # color = (0, 0, 0) # Green color in BGR
    # thickness = 2

    # mode_changer = StateChanger()
    # for i, r in enumerate(results):
    #     # StateChanger currently adapts only to results provided by YOLO
    #     mode_changer.record_detection(r.boxes.cls.tolist(), r.boxes.xywhn.tolist())
    #     mode_changer.change_state()
    #     cur_state = mode_changer._get_state()

    #     img = r.plot()
    #     cv2.putText(img, cur_state.value['mode'], coordinates, font, fontScale, outline_color, thickness + 4, cv2.LINE_AA)
    #     cv2.putText(img, cur_state.value['mode'], coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    #     cv2.imshow('bruh', img)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    #     if len(r.boxes.cls) > max:
    #         res = r
    #         max = len(r.boxes.cls)
    #     time.sleep(0.03)

    # cv2.destroyAllWindows()


# DEBUGGER
    # tmp = StateChanger()
    # tmp.record_detection([6], [[0.5, 0.5, 0.3, 0.9]])
    # tmp.change_state()
    # print(tmp._get_state().value['mode'])