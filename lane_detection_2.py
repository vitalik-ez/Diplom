import numpy as np
import matplotlib.pyplot as plt 
import cv2
import numpy as np


def binary_array(array, thresh, value=0):
    if value == 0:
        binary = np.ones_like(array)
    else:
        binary = np.zeros_like(array)
        value = 1
    binary[(array >= thresh[0]) & (array <= thresh[1])] = value
    return binary


def blur_gaussian(channel, ksize=3):
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
    sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return binary_array(mag, thresh)


def sobel(img_channel, orient='x', sobel_kernel=3):
    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)
    return sobel


def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)


#filename = '/content/threshold.jpg'
filename = '/content/test_german.png'


class Lane:

    def __init__(self, orig_frame):
        self.orig_frame = orig_frame
        self.lane_line_markings = None
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        length_line = self.height - 250
        self.roi_points = np.float32([
            (550, 540),  # Top-left corner
            (100, 720),  # Bottom-left corner
            (1050, 720),  # Bottom-right corner
            (650, 540),  # Top-right corner
        ])

        '''
        (490, 450),  # Top-left corner
            (110, 720),  # Bottom-left corner
            (1230, 720),  # Bottom-right corner
        (800, 450),  # Top-right corner
        '''



        '''
        self.roi_points = np.float32([
        (274,184), # Top-left corner
        (0, 337), # Bottom-left corner            
        (575,337), # Bottom-right corner
        (371,184) # Top-right corner
        ])
        '''

        self.padding = int(0.25 * width)
        self.desired_roi_points = np.float32([
            [self.padding, 0],  # Top-left corner
            [self.padding, self.orig_image_size[1]],  # Bottom-left corner
            [self.orig_image_size[
                0]-self.padding, self.orig_image_size[1]],  # Bottom-right corner
            [self.orig_image_size[0]-self.padding, 0]  # Top-right corner
        ])

        self.histogram = None


        self.no_of_windows = 10
        self.margin = int((1/12) * width)  
        self.minpix = int((1/24) * width)

        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = 10.0 / 1000  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 781  # meters per pixel in x dimension

        # Radii of curvature and offset
        self.left_curvem = None
        self.right_curvem = None
        self.center_offset = None

    def calculate_car_position(self, print_to_terminal=False):

        car_location = self.orig_frame.shape[1] / 2

        # Fine the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0]*height**2 + self.left_fit[
            1]*height + self.left_fit[2]
        bottom_right = self.right_fit[0]*height**2 + self.right_fit[
            1]*height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left)/2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(
            center_lane)) * self.XM_PER_PIX * 100

        if print_to_terminal == True:
            print(str(center_offset) + 'cm')

        self.center_offset = center_offset

        return center_offset

    def calculate_curvature(self, print_to_terminal=False):
        y_eval = np.max(self.ploty)

        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
            self.XM_PER_PIX), 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
            self.XM_PER_PIX), 2)

        # Calculate the radii of curvature
        left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[
                        1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvem = ((1 + (2*right_fit_cr[
                        0]*y_eval*self.YM_PER_PIX + right_fit_cr[
            1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Display on terminal window
        if print_to_terminal == True:
            print(left_curvem, 'm', right_curvem, 'm')

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

    def calculate_histogram(self, frame=None, plot=True):
        if frame is None:
            frame = self.warped_frame

        # Generate the histogram
        self.histogram = np.sum(frame[int(
            frame.shape[0]/2):, :], axis=0)

        if plot == True:

            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def display_curvature_offset(self, frame=None, plot=False):
        image_copy = None
        if frame is None:
            image_copy = self.orig_frame.copy()
        else:
            image_copy = frame

        cv2.putText(image_copy, 'Curve Radius: '+str((
            self.left_curvem+self.right_curvem)/2)[:7]+' m', (int((
                5/600)*self.width), int((
                    20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
                        0.5/600)*self.width)), (
            255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_copy, 'Center Offset: '+str(
            self.center_offset)[:7]+' cm', (int((
                5/600)*self.width), int((
                    40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
                        0.5/600)*self.width)), (
            255, 255, 255), 2, cv2.LINE_AA)

        return image_copy

    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        margin = self.margin

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store left and right lane pixel indices
        left_lane_inds = ((nonzerox > (left_fit[0]*(
            nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0]*(
                nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(
            nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0]*(
                nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        # Get the left and right lane line pixel locations
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        # Fit a second order polynomial curve to each lane line
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit
        self.right_fit = right_fit

        # Create the x and y values to plot on the image
        ploty = np.linspace(
            0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot == True:

            # Generate images to draw on
            out_img = np.dstack((self.warped_frame, self.warped_frame, (
                                 self.warped_frame)))*255
            window_img = np.zeros_like(out_img)

            # Add color to the left and right line pixels
            out_img[nonzeroy[left_lane_inds],
                    nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]
            # Create a polygon to show the search window area, and recast
            # the x and y points into a usable format for cv2.fillPoly()
            margin = self.margin
            left_line_window1 = np.array([np.transpose(np.vstack([
                                          left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                          left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([
                                           right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                           right_fitx+margin, ploty])))])
            right_line_pts = np.hstack(
                (right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def get_lane_line_indices_sliding_windows(self, plot=False):
        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        # Set the height of the sliding windows
        window_height = np.int(self.warped_frame.shape[0]/self.no_of_windows)

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store the pixel indices for the left and right lane lines
        left_lane_inds = []
        right_lane_inds = []

        # Current positions for pixel indices for each window,
        # which we will continue to update
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Go through one window at a time
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped_frame.shape[0] - \
                (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
                win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
                win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract the pixel coordinates for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial curve to the pixel coordinates for
        # the left and right lane lines
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        if plot == True:

            # Create the x and y values to plot on the image
            ploty = np.linspace(
                0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + \
                right_fit[1]*ploty + right_fit[2]

            # Generate an image to visualize the result
            out_img = np.dstack((
                frame_sliding_window, frame_sliding_window, (
                    frame_sliding_window))) * 255

            # Add color to the left line pixels and right line pixels
            out_img[nonzeroy[left_lane_inds],
                    nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]

            # Plot the figure with the sliding windows
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame with Sliding Windows")
            ax3.set_title("Detected Lane Lines with Sliding Windows")
            plt.show()

        return self.left_fit, self.right_fit

    def get_line_markings(self, frame=None):
        if frame is None:
            frame = self.orig_frame

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        _, sxbinary = threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = blur_gaussian(sxbinary, ksize=3)  # Reduce noise

        # 1s will be in the cells with the highest Sobel derivative values
        # (i.e. strongest lane line edges)
        sxbinary = mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = threshold(s_channel, (80, 255))

        _, r_thresh = threshold(frame[:, :, 2], thresh=(120, 255))

        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
            np.uint8))
        return self.lane_line_markings

    def histogram_peak(self):
        midpoint = np.int(self.histogram.shape[0]/2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # (x coordinate of left peak, x coordinate of right peak)
        return leftx_base, rightx_base

    def overlay_lane_lines(self, plot=False):
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
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                      self.orig_frame.shape[
                                          1], self.orig_frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if plot == True:

            # Plot the figures
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")
            ax2.set_title("Original Frame With Lane Overlay")
            plt.show()

        return result

    def perspective_transform(self, frame=None, plot=False):

        if frame is None:
            grayImage = cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
            frame = blackAndWhiteImage
            

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

        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                self.desired_roi_points]), True, (147, 20, 255), 3)

            cv2.destroyAllWindows()

        return self.warped_frame

    def plot_roi(self, frame=None, plot=False):
        if plot == False:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        # Overlay trapezoid on the frame
        this_image = cv2.polylines(frame, np.int32([
            self.roi_points]), True, (147, 20, 255), 3)

        # Display the image

        cv2.destroyAllWindows()


def detect(original_frame):

    lane_obj = Lane(orig_frame=original_frame)
    #return cv2.polylines(original_frame, np.int32([lane_obj.roi_points]), True, (147, 20, 255), 3)
    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()

    # Plot the region of interest on the image
    lane_obj.plot_roi(plot=False)

    # Perform the perspective transform to generate a bird's eye view
    # If Plot == True, show image with new region of interest
    warped_frame = lane_obj.perspective_transform(plot=False)
    #return warped_frame
    # Generate the image histogram to serve as a starting point
    # for finding lane line pixels
    histogram = lane_obj.calculate_histogram(plot=False)

    # Find lane line pixels using the sliding window method
    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
        plot=False)

    # Fill in the lane line
    lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)

    # Overlay lines on the original frame
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
    return cv2.cvtColor(frame_with_lane_lines, cv2.COLOR_BGR2RGB)
    #cv2_imshow(frame_with_lane_lines)
    # Calculate lane line curvature (left and right lane lines)
    lane_obj.calculate_curvature(print_to_terminal=False)

    # Calculate center offset
    lane_obj.calculate_car_position(print_to_terminal=False)

    # Display curvature and center offset on image
    frame_with_lane_lines2 = lane_obj.display_curvature_offset(
        frame=frame_with_lane_lines, plot=True)
    
    # Create the output file name by removing the '.jpg' part
    size = len(filename)
    new_filename = filename[:size - 4]
    new_filename = new_filename + '_thresholded.jpg'

    # Save the new image in the working directory
    #cv2.imwrite(new_filename, lane_line_markings)

    # Display the image
    #cv2.imshow("Image", lane_line_markings)

    # Display the window until any key is pressed
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()



if __name__ == '__main__':
    detect()
