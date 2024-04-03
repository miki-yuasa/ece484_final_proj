import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO
        Img_height = img.shape[0]
        Img_width = img.shape[1]
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        after_blur = cv2.GaussianBlur(grey, (5,5), 0)
        dx = cv2.Sobel(after_blur, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(after_blur, cv2.CV_32F, 0, 1, ksize=3)
        abs_dx = np.absolute(dx)
        abs_dy = np.absolute(dy)        
        result_combined = cv2.addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0)
        after_scale = np.uint8(result_combined)
        binary_output = np.zeros_like(after_scale)
        binary_output[(after_scale >= thresh_min) & (after_scale <= thresh_max)] = 255
        # cv2.imshow("Sober Combined", after_scale)
        # cv2.waitKey(0)

        ####

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO
        Img_height = img.shape[0]
        Img_width = img.shape[1]
        convert_to_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        h_cha = convert_to_hls[:,:,0]
        l_cha = convert_to_hls[:,:,1]
        s_cha = convert_to_hls[:,:,2]

        yellow_binary = np.zeros_like(h_cha)
        white_binary = np.zeros_like(s_cha)
        # yellow_binary[(h_cha >= 82) & (h_cha <= 105) & (s_cha >= 0.25*255) & (s_cha <= 1*255)] = 255
        white_binary[(l_cha >= 255*0.82) & (l_cha <= 255) & (s_cha >= 0) & (s_cha <= 0.3*255)] = 255
        yellow_binary[(h_cha >= 82) & (h_cha <= 105) & (s_cha >= 0.25*255) & (s_cha <= 1*255)] = 255

        binary_output = np.zeros_like(white_binary)
        
        binary_output[white_binary == 255|yellow_binary == 255] = 255
        binary_output=np.uint8(binary_output)
        ####
        # cv2.imshow("color Combined", binary_output)
        # cv2.waitKey(0)
        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        sober_output = self.gradient_thresh(img)
        color_output = self.color_thresh(img)
        ####
        
        binaryImage = np.zeros_like(sober_output)
        binaryImage[(color_output==255)|(sober_output==255)] = 255
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        binaryImage = np.uint8(binaryImage)
        # cv2.imshow("color Combined", binaryImage)
        # cv2.waitKey(0)
        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        # For gazebo images
        pts1 = np.float32([[60, 360], [146, 320], [587, 360], [500, 320]])
        pts2 = np.float32([[100, 390], [0, 360], [560, 390], [560, 360]])

        img_height = img.shape[0]
        img_width = img.shape[1]

        # For rosbag images in 375 x 1242 resolution
        # pts1 = np.float32(np.float32([[400, 320], [470, 280], [780, 320], [742, 280]])*np.array([img_width/1242, img_height/375]))
        # pts2 = np.float32(np.float32([[350, 330], [350, 260], [892, 330], [892, 260]])*np.array([img_width/1242, img_height/375]))

        M = cv2.getPerspectiveTransform(pts1, pts2)
        Minv = cv2.getPerspectiveTransform(pts2, pts1)

        warped_img = cv2.warpPerspective(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR
        )
        ####

        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
