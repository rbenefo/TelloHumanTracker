import numpy as np
import cv2

from personDetector import PersonDetector
import matplotlib.pyplot as plt


class PersonTracker:
    score_threshold = 0.5
    color_dist_threshold = 40
    def __init__(self):
        self.personDetector = PersonDetector('ssd_mobilenet_v1_coco_2017_11_17')
        self.imprinted = False
        self.r = None
        self.g = None
        self.b = None
        # self.i = 0

    def findMyHooman(self, image):
        '''Find human by detecting largest human bounding box. 
        Maybe later, figure out how to detect one that's moved the least?'''
        # self.i += 1
        boxes, scores = self.personDetector.infer(image)
        ind = np.where(scores > self.score_threshold)
        thresholdedBoxes = boxes[ind]
        if thresholdedBoxes.size != 4:
            return None, self.imprinted
        
        boxSizes = (thresholdedBoxes[:,0]-thresholdedBoxes[:,2])**2+(thresholdedBoxes[:,1]-thresholdedBoxes[:,3])**2

        if boxSizes.size == 0:
            return None, self.imprinted
        else:
            if not self.imprinted and boxSizes.size == 1:
                largestBox = thresholdedBoxes[np.argmax(boxSizes)]
                masked = image[int(largestBox[0]*image.shape[0]):int(largestBox[2]*image.shape[0]),\
                    int(largestBox[1]*image.shape[1]):int(largestBox[3]*image.shape[1])]
                self.r,self.g,self.b = np.mean(masked[:,:,0]), np.mean(masked[:,:,1]), np.mean(masked[:,:,2])
                self.imprinted = True
                print("IMPRINTED.")
                return largestBox, self.imprinted
            elif self.imprinted and boxSizes.size == 1:
                largestBox = thresholdedBoxes[np.argmax(boxSizes)]
                masked = image[int(largestBox[0]*image.shape[0]):int(largestBox[2]*image.shape[0]),\
                    int(largestBox[1]*image.shape[1]):int(largestBox[3]*image.shape[1])]
                r, g, b = np.mean(masked[:,:,0]), np.mean(masked[:,:,1]), np.mean(masked[:,:,2])
                dist = np.abs(r-self.r)+np.abs(g-self.g)+np.abs(b-self.b)
                # plt.scatter(self.i, dist)
                # self.i += 1
                # plt.pause(0.05)
                # if self.i >= 2400:
                #     plt.show()

                # print("dist: {}".format(dist))
                if dist > self.color_dist_threshold:
                    # print("NOPE! GHOSTS ABOUND.")
                    return None, self.imprinted
                else:
                    # print("EHH, CLOSE ENOUGH.")
                    return largestBox, self.imprinted
            elif self.imprinted:
                ind = np.argpartition(boxSizes, -2)[-2:] #get largest 2 indicies
                dist = 10000 #initialize to large number
                sorted_inds = ind[np.argsort(boxSizes[ind])]
                selected_index = sorted_inds[0]
                for index in sorted_inds:
                    box = thresholdedBoxes[index]
                    masked = image[int(box[0]*image.shape[0]):int(box[2]*image.shape[0]), \
                        int(box[1]*image.shape[1]):int(box[3]*image.shape[1])]
                    r, g, b = np.mean(masked[:,:,0]), np.mean(masked[:,:,1]), np.mean(masked[:,:,2])
                    new_dist = np.abs(r-self.r)+np.abs(g-self.g)+np.abs(b-self.b)
                    if new_dist < dist:
                        dist = new_dist
                        selected_index = index  
                return thresholdedBoxes[selected_index], self.imprinted
            else:
                return None, self.imprinted


if __name__ =="__main__":
    personTracker = PersonTracker()
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        humanBox = personTracker.findMyHooman(frame)
        if humanBox is not None:
            xmin = int(humanBox[1]*frame.shape[1])
            ymin = int(humanBox[0]*frame.shape[0])
            xmax = int(humanBox[3]*frame.shape[1])
            ymax = int(humanBox[2]*frame.shape[0])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2) #blue

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


        