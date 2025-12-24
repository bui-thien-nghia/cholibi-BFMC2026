from src.templates.workerprocess import WorkerProcess
from src.perception.lanedetection.threads import threadLaneDetection

class processLaneDetection(WorkerProcess):
    def __init__(self, queuesList, logging, ready_event=None, debugging=False):
        self.queuesList = queuesList
        self.logging = logging
        self.debugging = debugging

        super(processLaneDetection, self).__init__(self.queuesList, ready_event)

    
    def _init_threads(self):
        laneTh = threadLaneDetection(
            self.queuesList, self.logging, self.debugging
        )
        self.threads.append(laneTh)