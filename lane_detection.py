import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection:

    def __init__(self) -> None:
        self.height = 720
        self.width = 1280

        #self.orig_frame = orig_frame

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = (self.width, self.height)#self.orig_frame.shape[::-1][1:]

    def sobel_edge_detection(self, frame, kernel=3):
        sobelx = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=kernel)
        sobely = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=kernel)
        #sobelxy = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=kernel) # Combined X and Y Sobel Edge Detection
        absolute_x = np.absolute(sobelx)
        absolute_y = np.absolute(sobely)
        sobelxy = np.sqrt(absolute_x ** 2 + absolute_y ** 2)
        return sobelxy
    
    def threshold(self, channel, thresh, maxValue):
        return cv2.threshold(channel, thresh, maxValue, cv2.THRESH_BINARY)

    def preprocessing(self):
        """
        Isolates lane lines.

          :param frame: The camera frame that contains the lanes we want to detect
        :return: Binary (i.e. black and white) image containing the lane lines.
        """

        hls = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS)
        _, binary = self.threshold(hls[:, :, 1], thresh=120, maxValue=255)
        
        ksize = 3
        binary = cv2.GaussianBlur(binary, (ksize, ksize), 0)  # Reduce noise

        sobel = self.sobel_edge_detection(binary, kernel=3)
        thresh=(110, 255)
        
        binary = np.ones_like(sobel)
        binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 0
        
        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = self.threshold(s_channel, thresh=80, maxValue=255)

        _, r_thresh = self.threshold(self.frame[:, :, 2], thresh=120, maxValue=255)

        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        lane_line_markings = cv2.bitwise_or(rs_binary, binary.astype(
            np.uint8))

        return lane_line_markings

    def perspective_transform(self, frame):
        length_line = self.height - 250
        #orig_img_coord = np.float32([[490, 450], [200, 720],  [800, length_line], [1230, self.height]])
        self.roi_points = np.float32([(490, 450), (110, 720),(1230, 720), (800, 450)])

        #for x in range(4):
        #    image = cv2.circle(image, (int(orig_img_coord[x][0]), int(orig_img_coord[x][1])), 10, (0,0,255), -1)
        #cv2_imshow(image)
        height, width = 350, 450

        self.padding = int(0.25 * width) # padding from side of the image in pixels
        self.desired_roi_points = np.float32([
          [self.padding, 0], # Top-left corner
          [self.padding, self.orig_image_size[1]], # Bottom-left corner         
          [self.orig_image_size[
            0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
          [self.orig_image_size[0]-self.padding, 0] # Top-right corner
        ]) 

        perspective_coordinates = np.float32([[self.padding, 0], 
                                              [self.padding, self.width], 
                                              [self.height-self.padding, self.width], 
                                              [self.height-self.padding, 0]])

        '''
         # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
          self.roi_points, self.desired_roi_points)
    
        # Calculate the inverse transformation matrix           
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
          self.desired_roi_points, self.roi_points)
    
        # Perform the transform using the transformation matrix
        self.warped_frame = cv2.warpPerspective(
          frame, self.transformation_matrix, self.orig_image_size, flags=(
        cv2.INTER_LINEAR)) 
        '''
        matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)
        transform_img = cv2.warpPerspective(frame, matrix, self.orig_image_size, flags=(cv2.INTER_LINEAR))
        
        self.inverse_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)
        return transform_img

    def sliding_windows(self, warped_frame):
        #cv2.imshow("perspective_transform", warped_frame)
        #cv2.imshow("bottom_half_perspective_transform", warped_frame[warped_frame.shape[0]//2:, :])
        histogram = np.sum(warped_frame[warped_frame.shape[0]//2:, :], axis=0)

        out_img = np.dstack((warped_frame, warped_frame, warped_frame))

        mid_point = histogram.shape[0]//2
        left_line_base_index = np.argmax(histogram[:mid_point])
        right_line_base_index = np.argmax(histogram[mid_point:]) + mid_point
        
        num_windows = 10
        margin = 20
        minimim_pixels = 25

        height_window = np.int8(warped_frame.shape[0] // num_windows)
        print(height_window)

        non_zeroy, non_zerox = warped_frame.nonzero()

        # Current base position on x-axis (+/- margin) 
        left_line_current_index = left_line_base_index
        right_line_current_index = right_line_base_index

        left_lane = []
        right_lane = []

        for window in range(num_windows): 
            
            # Coordinates left line sliding box
            left_line_y_1 = warped_frame.shape[0] - height_window * (window + 1)
            left_line_y_2 = warped_frame.shape[0] - height_window * window
            
            left_line_x_1 = left_line_current_index - margin
            left_line_x_2 = left_line_current_index + margin

            # Coordinates right line sliding box
            right_line_y_1 = warped_frame.shape[0] - height_window * (window + 1)
            right_line_y_2 = warped_frame.shape[0] - height_window * window
            
            right_line_x_1 = right_line_current_index - margin
            right_line_x_2 = right_line_current_index + margin


            # Draw boxes
            #warped_frame = cv2.rectangle(warped_frame,(left_line_x_1, left_line_y_1),(left_line_x_2,left_line_y_2),(255,255,255),2)
            #warped_frame = cv2.rectangle(warped_frame,(right_line_x_1, right_line_y_1),(right_line_x_2,right_line_y_2),(255,255,255),2)
            #cv2.imshow("result", warped_frame)


            # Identify the nonzero pixels in x and y within the window
            nonzero_left_sliding_box = ((non_zeroy >= left_line_y_1) & (non_zeroy < left_line_y_2) &
                              (non_zerox >= left_line_x_1) & (non_zerox < left_line_x_2)).nonzero()[0]
            
            nonzero_right_sliding_box = ((non_zeroy >= right_line_y_1) & (non_zeroy < right_line_y_2) &
                              (non_zerox >= right_line_x_1) & (non_zerox < right_line_x_2)).nonzero()[0]
            
            left_lane.append(nonzero_left_sliding_box)
            right_lane.append(nonzero_right_sliding_box)
            

            if len(nonzero_left_sliding_box) > minimim_pixels:
                left_line_current_index = int(np.mean(non_zerox[nonzero_left_sliding_box]))
            if len(nonzero_right_sliding_box) > minimim_pixels:        
                right_line_current_index = int(np.mean(non_zerox[nonzero_right_sliding_box]))


        try:
            left_lane = np.concatenate(left_lane)
            right_lane = np.concatenate(right_lane)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        

        leftx = non_zerox[left_lane]
        lefty = non_zeroy[left_lane] 
        rightx = non_zerox[right_lane]
        righty = non_zeroy[right_lane]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        x_coordinates = np.linspace(0, warped_frame.shape[0]-1, warped_frame.shape[0])
        left_fitx = left_fit[0]*x_coordinates**2 + left_fit[1]*x_coordinates + left_fit[2]
        right_fitx = right_fit[0]*x_coordinates**2 + right_fit[1]*x_coordinates + right_fit[2]
        
        self.ploty = x_coordinates
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

    
    def overlay_lane_lines(self, warped_frame, plot=False):
        """
        Overlay lane lines on the original frame
        :param: Plot the lane lines if True
        :return: Lane with overlay
        """
        # Generate an image to draw the lane lines on
        self.warped_frame = warped_frame
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([
                             self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
                              self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inverse_matrix, (self.frame.shape[1], self.frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(self.frame, 1, newwarp, 0.3, 0)

        return result


    def detect(self, frame):
        self.frame = frame #cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_AREA)

        frame = self.preprocessing()

        warped_frame = self.perspective_transform(frame)
        #return frame, warped_frame
        #cv2.imshow("orig_frame", lane_detection.frame)
        #cv2.imshow("perspective_transform", warped_frame)

        self.sliding_windows(warped_frame)
        result = self.overlay_lane_lines(warped_frame)

        return result

    





if __name__ == '__main__':
    lane_detection = LaneDetection()

    frame = cv2.imread("test_data/test.png")

    frame, warped_frame = lane_detection.detect(frame=frame)

    cv2.imshow("orig_frame", frame)
    cv2.imshow("perspective_transform", warped_frame)
