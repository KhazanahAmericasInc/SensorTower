import cv2
import os
import sys
import io 
import picamera
import time
import datetime as dt
import numpy as np
from queue import Queue
from threading import Thread

#TODO mask threshold and velocity
MASK_THRESHOLD = 180
BOX_THRESHOLD = 6000
IMG_WIDTH = 360
IMG_HEIGHT = 240
VELOCITY = 50

image_queue = Queue(maxsize=0)
filepath = '/home/pi/Tracker/'

def dropbox_upload():
    #print ("Before img conter")
    #image_counter = 1
    while True:
        #print ("Inside While")
        crop_img,direction = image_queue.get()
        if crop_img is not None:
            #print ("Inside If")
            imgName = "{0}_{1}.png".format(dt.datetime.now().strftime('%m-%d_%H.%M.%S.%f'),direction)
            cv2.imwrite(imgName,crop_img)
            upload = '/home/pi/Dropbox-Uploader/dropbox_uploader.sh upload ' + imgName + ' /Sensor_Tower_Final_2018/'
            os.system(upload)
            os.remove(imgName)
            #image_counter += 1

def save(bbox, img, save_counter):
    crop_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    cv2.imwrite("image{}.png".format(save_counter), crop_img)

class Solution(object):
    def __init__(self, mask_threshold = MASK_THRESHOLD, box_threshold = BOX_THRESHOLD):
        self.mask_threshold = mask_threshold
        self.left_counter = 0
        self.right_counter = 0

        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        #self.fgbg.setHistory(75)
        self.fgbg.setVarThreshold(25)

        self.box_threshold = box_threshold

    def print_counter(self):
        print("LEFT: {0}, RIGHT: {1}".format(self.left_counter, self.right_counter))

    def initialize_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        
        #params.minThreshold = 10
        #params.maxThreshold = 200

        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 8000
        
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False

        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def find_mask(self, img):
        #Background subtractor
        fgmask = self.fgbg.apply(img)

        #initial blur
        blurred = cv2.blur(fgmask, (5,5))
        #blurred = cv2.medianBlur(fgmask, 3)

        #Binary thresholding
        _, thresh = cv2.threshold(blurred, self.mask_threshold, 255, cv2.THRESH_BINARY_INV)
        
        #Erode image
        erode_kernel = np.ones((3,3), np.uint8)
        thresh = cv2.erode(thresh, erode_kernel, iterations=3)

        #Median Blur
        thresh = cv2.medianBlur(thresh, 7)
        #thresh = cv2.medianBlur(thresh, 7)

        #flood filling
        #thresh_fill = thresh.copy()
        #mask = np.zeros((IMG_HEIGHT + 2, IMG_WIDTH + 2), np.uint8)

        #cv2.floodFill(thresh_fill, mask, (0, 0), 255)
        #out = thresh_fill | thresh

        return fgmask, thresh

    def find_contours(self, img):
        _, mask = self.find_mask(img)
        mask = cv2.bitwise_not(mask)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return mask, contours

    def find_bounding_boxes(self, img):
        mask, contours = self.find_contours(img)
        #epsilon = [0.1*cv2.arcLength(contour, True) for contour in contours]
        #approx_poly = [cv2.approxPolyDP(contour, epsilon, True) for contour, epsilon in zip(contours, epsilon)]
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        filtered_boxes = []
        for bbox in bounding_boxes:
            if (bbox[2]*bbox[3] > self.box_threshold): # and (bbox[2] * 2 > bbox[3]):
                filtered_boxes.append(bbox)
        
        return mask, filtered_boxes

    def blob_detect(self, img):
        _, mask = self.find_mask(img)
        #Pad and blob detect
        padded_img = cv2.copyMakeBorder(mask, 3, 3, 3, 3, cv2.BORDER_CONSTANT, None, 255)
        keypoints = self.blob_detector.detect(padded_img)
        
        return mask, keypoints

    def image_count(self):
        return self.right_counter + self.left_counter

'''
class VerticalLineSolution(Solution):
    def __init__(self, mask_threshold = MASK_THRESHOLD):
        Solution.__init__(self, mask_threshold)
        self.left_lines = None
        self.right_lines = None
        self.left_enter = False
        self.left_exit = False
        self.right_enter = False
        self.right_exit = False
'''

class HorizontalLineSolution(Solution):
    def __init__(self, mask_threshold = MASK_THRESHOLD):
        Solution.__init__(self, mask_threshold)
        self.photo_line = 150
        self.split_line = 120
        self.photo_threshold = 5
        self.dir1_on = False # Makes it so that cars that are in the threshold for
        self.dir2_on = False # multiple frame doesnt get captured multiple times
        

    def process(self, img):
        mask, bboxes = self.find_bounding_boxes(img)
        no_dir1 = True
        no_dir2 = True
        for bbox in bboxes:
            if abs(bbox[0] + bbox[2]/2 - self.photo_line) < self.photo_threshold:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 4)
               #take image
                if bbox[1] + bbox[3] > self.split_line: 
                    no_dir1 = False
                    if not self.dir1_on:
                        self.left_counter += 1
                        self.dir1_on = True
                else:
                    no_dir2 = False
                    if not self.dir2_on:
                        self.right_counter += 1
                        self.dir2_on = True

        if no_dir1:
            self.dir1_on = False
        if no_dir2:
            self.dir2_on = False

        #show lines
        #cv2.line(img, (0, self.split_line), (300, self.split_line), (255, 0, 0), 4) 
        #cv2.line(img, (self.photo_line, 0), (self.photo_line, 300), (255, 0, 0), 4) 

        return img

class KCFTracker(object):
    def __init__(self, bbox, direction, img):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(img, bbox)
        self.bbox = bbox
        self.undetected = 0
        self.direction = direction  #right = 0, left = 1
        self.x, self.y = self.get_center()
        self.age = 0

    def get_center(self):
        x, y = self.bbox[0] + self.bbox[2]/2, self.bbox[1] + self.bbox[3]/2
        return x, y
    def predict(self):
        if self.direction == 0:
            self.bbox = self.bbox[0] + VELOCITY, self.bbox[1], self.bbox[2], self.bbox[3] 
        else:
            self.bbox = self.bbox[0] - VELOCITY, self.bbox[1], self.bbox[2], self.bbox[3] 

    def update(self, img):
        if self.undetected != 0:
            self.predict()
        else:
            ok, bbox = self.tracker.update(img)
            if not ok:
                #print("UNDETECTED")
                self.undetected = 1
                self.predict()
            else:
                self.bbox = bbox
        self.x, self.y = self.get_center()
        self.age += 1

    def bbox_points(self): 
        p1 = (int(self.bbox[0]), int(self.bbox[1]))
        p2 = (int(self.bbox[0]+self.bbox[2]), int(self.bbox[1]+self.bbox[3]))
        return p1, p2

    def in_bbox(self, bbox):
        in_x = bbox[0] < self.x and bbox[0]+bbox[2] > self.x
        in_y = bbox[1] < self.y and bbox[1]+bbox[3] > self.y 

        return in_x and in_y

    def inside_image(self):
        return self.in_bbox((0, 0, IMG_WIDTH, IMG_HEIGHT))

    def calc_dist(self, bbox):
        dist_x = abs(self.bbox[0] + self.bbox[2]/2 - bbox[0] - bbox[2]/2)
        dist_y = abs(self.bbox[1] + self.bbox[3]/2 - bbox[1] - bbox[3]/2)
        
        return dist_x + dist_y

    def reinitialize(self, img, bbox):
        #print ("REINIT")
        self.tracker.clear() 
        self.tracker = cv2.TrackerKCF_create()
        if self.direction == 1:
            self.bbox = bbox[0] - 5, bbox[1], bbox[2], bbox[3]
        else:
            self.bbox = bbox[0] + 5, bbox[1], bbox[2], bbox[3]
        self.tracker.init(img, self.bbox)
        self.undetected = 0
    
class KCFTrackerSolution(Solution):
    def __init__(self, mask_threshold = MASK_THRESHOLD):
        Solution.__init__(self, mask_threshold)
        self.tracker_list = []
        self.age_threshold = 20

    def update_trackers(self, img):
        for tracker in self.tracker_list:
            tracker.update(img)

        self.tracker_list[:] = [tracker for tracker in self.tracker_list if tracker.inside_image() and tracker.age < self.age_threshold]

    def add_trackers(self, img, bboxes):
        for bbox in bboxes:
            if bbox[0] > 10 and bbox[0] + bbox[2] < IMG_WIDTH - 10:
                found = False
                reinit = None
                for tracker in self.tracker_list:
                    if tracker.in_bbox(bbox):
                        if tracker.undetected == 1:
                            reinit = tracker
                        else:
                            found = True
                            break
                if not found:
                    if reinit:
                        reinit.reinitialize(img, bbox)
                    else:
                        if bbox[0] > 50:
                            bbox_fixed = bbox[0] - 5, bbox[1], bbox[2], bbox[3]
                            kcftracker = KCFTracker(bbox_fixed, 1, img)
                            self.tracker_list.append(kcftracker)
                            self.left_counter += 1
                            crop_img = img[bbox_fixed[1]:bbox_fixed[1]+bbox_fixed[3], bbox_fixed[0]:bbox_fixed[0]+bbox_fixed[2]].copy()
                            image_queue.put((crop_img, "LEFT"))
                        else:
                            bbox_fixed = bbox[0] + 5, bbox[1], bbox[2], bbox[3]
                            kcftracker = KCFTracker(bbox_fixed, 0, img)
                            self.tracker_list.append(kcftracker)
                            self.right_counter += 1
                            crop_img = img[bbox_fixed[1]:bbox_fixed[1]+bbox_fixed[3], bbox_fixed[0]:bbox_fixed[0]+bbox_fixed[2]].copy()
                            image_queue.put((crop_img, "RIGHT"))
                        print(image_queue.qsize())
                        self.print_counter()
                        #cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 5)

    def draw_trackers(self, img):
        for tracker in self.tracker_list:
            #if (tracker.undetected == 0):
                p1, p2 = tracker.bbox_points()
                cv2.rectangle(img, p1, p2, (0, 0, 255), 1)
                #print(tracker.get_center())

    def process(self, img):
        _, bboxes = self.find_bounding_boxes(img)
        #print (len(bboxes))
        self.update_trackers(img)
        self.add_trackers(img, bboxes)
        #self.draw_trackers(img)
        #print (len(self.tracker_list))
        return img

def main():
    uploader = Thread(target=dropbox_upload)
    uploader.start()
    #stream = io.BytesIO()
    sol = KCFTrackerSolution()
    top_crop = 10
    bot_crop = 10
    with picamera.PiCamera() as camera:
        #stream = io.BytesIO()
        camera.rotation =180
        #camera.start_preview()
       	#time.sleep(2)
        counter = 0
        start = time.time()
        while (1):
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg',  resize=(IMG_WIDTH, IMG_HEIGHT), use_video_port = True)
            # Construct a numpy array from the stream
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            # "Decode" the image from the array, preserving colour
            image = cv2.imdecode(data, 1)
            # OpenCV returns an array with data in BGR order. If you want RGB instead
            # use the following...
            #image = image[:, :, ::-1]
            crop_img = image[top_crop:IMG_HEIGHT - bot_crop, 0:IMG_WIDTH]
            sol.process(image)
            #cv2.imwrite("test.png", image)
            end = time.time()
            print (end - start)
            start = end
            counter += 1
            

if __name__ == "__main__":
    main()
