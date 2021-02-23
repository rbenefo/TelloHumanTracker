from personTracker import PersonTracker
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tello import Tello # pylint: disable=import-error
import pygame
import numpy as np
import time
import cv2


# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

class Follower:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.tello = Tello()
        self.send_rc_control = False
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
        
        self.personTracker = PersonTracker()

        self.Px, self.Ix, self.Dx = 0.10,0,-0 #D gain should be negative.
        self.Py, self.Iy, self.Dy = 0.1,0,-0
        self.Pz, self.Iz, self.Dz = 0.25,0,-0.001

        self.prev_err_x, self.prev_err_y, self.prev_err_z = None, None, None
        self.accum_err_x, self.accum_err_y, self.accum_err_z = 0,0,0
        self.found_person = False
        self.manual = True

        self.iter = 0



    def run(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        imprinted = False
        emergency_counter = 0
        while not should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)
            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            if self.iter > 240:
                humanBox, imprinted = self.personTracker.findMyHooman(frame)
                if humanBox is None and self.send_rc_control and not self.manual:
                    emergency_counter += 1
                    if emergency_counter >= 120: #missed human for 120 frames; 1 second
                        print("ENGAGING EMERGENCY HOVER.")
                        cv2.putText(frame, "EMERGENCY HOVER", (700, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        self.emergencyHover()
                elif humanBox is not None:
                    if self.found_person == False:
                        self.found_person = True
                    emergency_counter = 0

                #PID LOOP:
                desired_pos = (frame.shape[1]//2, frame.shape[0]//2) #format x,y

                if humanBox is not None:
                    xmin = int(humanBox[1]*frame.shape[1])
                    ymin = int(humanBox[0]*frame.shape[0])
                    xmax = int(humanBox[3]*frame.shape[1])
                    ymax = int(humanBox[2]*frame.shape[0])
                    centerHumanPosition = (np.mean((xmax, xmin)), np.mean((ymax, ymin))) #format x,y
                    #draw bounding box

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2) #blue
                    #draw target coord
                    
                    cv2.circle(frame, (int(centerHumanPosition[0]), int(centerHumanPosition[1])), 10, (255, 0,0), 1) #blue
                    # print("z width: {}".format(np.abs(xmax-xmin)))
                    #draw desired coord
                    cv2.circle(frame, desired_pos, 10, (0, 0,255), 1) #red


                    if self.send_rc_control and not self.manual:                
                        self.update_control(centerHumanPosition, desired_pos, xmax, xmin)


            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            heightText = "Height:{}".format(self.tello.get_height())

            cv2.putText(frame, heightText, (720-5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            manualText = "Manual: {}".format(self.manual)
            if self.manual:
                cv2.putText(frame, manualText, (720-5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, manualText, (720-5, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            imprintedTxt = "Imprinted: {}".format(imprinted)
            if imprinted:
                cv2.putText(frame, imprintedTxt, (720-5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, imprintedTxt, (720-5, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            if self.iter <= 240:
                self.iter +=1
            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_m:  # set yaw clockwise velocity
            self.manual = not self.manual
            self.yaw_velocity = 0
            self.up_down_velocity =0
            self.left_right_velocity =0
            print("MANUAL MODE IS NOW: {}".format(self.manual))


    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
            
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.iter = 0
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

    def update_control(self, curr_lateral_pos, desired_later_pos, xmax, xmin):
        #Three error directions. Two lateral, one forward/backward. How to calc forward/back error?
        err_x = curr_lateral_pos[0] - desired_later_pos[0] #if positive, we're to the left of where we want to be, so want positive control. (CHECK THIS)
        err_y = desired_later_pos[1]- curr_lateral_pos[1]#if positive, we're below where we want to be. (CHECK THIS)

        #hardcode desired box width. Must test!!
        desired_width = 350
        curr_width = np.abs(xmax-xmin) #check. is this actually the width dim?

        err_z = desired_width-curr_width #if negative, too close; want backwards--> positive gain
        # print("Err z: {}".format(err_z))

        if self.prev_err_x == None:
            derivative_x_input = 0
            derivative_y_input = 0
            derivative_z_input = 0
        else:
            derivative_x_input = (err_x - self.prev_err_x)/(1/FPS)
            derivative_y_input = (err_y - self.prev_err_y)/(1/FPS)
            derivative_z_input = (err_z - self.prev_err_z)/(1/FPS)

            #clip derivative errors to avoid noise
            derivative_x_input = np.clip(derivative_x_input, -11000, 11000)
            derivative_y_input = np.clip(derivative_y_input, -11000, 11000)
            derivative_z_input = np.clip(derivative_z_input, -11000, 11000)


        self.accum_err_x += err_x
        self.accum_err_y += err_y
        self.accum_err_z += err_z

        self.prev_err_x = err_x
        self.prev_err_y = err_y
        self.prev_err_z = err_z

        # print("derr_z: {}".format(derivative_z_input))



        # self.left_right_velocity = self.Px*err_x+self.Dx*derivative_x_input+self.Ix*self.accum_err_x
        self.yaw_velocity = self.Px*err_x+self.Dx*derivative_x_input+self.Ix*self.accum_err_x
        self.up_down_velocity = self.Py*err_y+self.Dy*derivative_y_input+self.Iy*self.accum_err_y
        self.for_back_velocity = self.Pz*err_z+self.Dy*derivative_z_input+self.Iz*self.accum_err_z

        #limit velocity to 2*S.
        # self.left_right_velocity = np.clip(self.left_right_velocity, -S*2, S*2)
        self.yaw_velocity = int(np.clip(self.yaw_velocity, -S, S))
        self.up_down_velocity = int(np.clip(self.up_down_velocity, -S, S))
        self.for_back_velocity = int(np.clip(self.for_back_velocity, -S, S))


        #Send new velocities to robot.
        self.update()

    
    def emergencyHover(self):
        print("Cannot find hooman. I am lonely doggo. Hovering and rotating.")

        self.found_person = False

        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = int(S//2)
        self.update()

def main():
    print("Running...")
    follower = Follower()

    # run follower
    follower.run()


if __name__ == '__main__':
    main()
