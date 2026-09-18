EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 10

class DrowsinessDetector:
    def __init__(self):
        self.counter = 0
        self.drowsy = False
    
    def check_drowsiness(self, ear):
        if ear < EAR_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0
        
        if self.counter >= CONSECUTIVE_FRAMES:
            self.drowsy = True
        else:
            self.drowsy = False
        
        return self.drowsy