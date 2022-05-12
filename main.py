import cv2
import numpy as np


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 50)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


class LaneDetection:

    def __init__(self) -> None:
        pass

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

    def preprocessing(self, frame):
        """
        Isolates lane lines.

          :param frame: The camera frame that contains the lanes we want to detect
        :return: Binary (i.e. black and white) image containing the lane lines.
        """

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        _, binary = self.threshold(hls[:, :, 1], thresh=120, maxValue=255)
        
        ksize = 3
        binary = cv2.GaussianBlur(binary, (ksize, ksize), 0)  # Reduce noise

        sobel = self.sobel_edge_detection(binary, kernel=3)
        thresh=(110, 255)
        
        binary = np.ones_like(sobel)
        binary[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 0
        
        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = self.threshold(s_channel, thresh=80, maxValue=255)

        _, r_thresh = self.threshold(frame[:, :, 2], thresh=120, maxValue=255)

        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        ### Combine the possible lane lines with the possible lane line edges #####
        # If you show rs_binary visually, you'll see that it is not that different
        # from this return value. The edges of lane lines are thin lines of pixels.
        lane_line_markings = cv2.bitwise_or(rs_binary, binary.astype(
            np.uint8))

        return lane_line_markings


if __name__ == '__main__':
    '''
    image = cv2.imread('test_data/test7.png')
    image = cv2.resize(image, (900,600), interpolation = cv2.INTER_AREA)
    lane_detection = LaneDetection()
    result = lane_detection.preprocessing(image)
    #cv2.imwrite('binary_image.png', result)
    '''
    

    lane_detection = LaneDetection()
    cap = cv2.VideoCapture("test_data/test3.mp4")
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    else:
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (900,600), interpolation = cv2.INTER_AREA)
                cv2.imshow("result", lane_detection.preprocessing(frame))
                cv2.waitKey(1)

                if cv2.waitKey(25) == ord('q'):
                    break
            else:
                print("Error reading frame")
                break


    cap.release()
    cv2.destroyAllWindows()
    


    '''
    cap = cv2.VideoCapture("test_data/test3.mp4")

    while(cap.isOpened()):
        _ , frame = cap.read()
        canny_image = canny(frame)
        cv2.imshow("canny", canny_image)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('result', combo_image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''









