import numpy as np
import cv2
import math

alt = 0.70  # Altitude of drone in meter
screen_width = 640
screen_height = 480
phai = 51.4  # Horizontal_FOV
gamma = 31.12  # Vertical_FOV

def convert_screen_to_real(dx, dy):
    xx = dx * 2 * alt * math.tan(math.radians(phai / 2)) / screen_width
    yy = dy * 2 * alt * math.tan(math.radians(gamma / 2)) / screen_height
    return xx, yy


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of each camera pane in the window on the screen
# Default 1920x1080 displayd in a 1/4 size window

def gstreamer_pipeline(
        sensor_id=0,
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
):
    return (
            "nvarguscamerasrc sensor-id=%d !"
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )

def Calculate_mean_by_distribution(x_dist, y_dist, t_dist):

    mean_t = t_dist.mean()

    #Calcluate standard deviation of distances
    count = 0
    sum = 0
    for item in t_dist:
        if item != 0:      #remove zero elements
            sum = sum + (item - mean_t) ** 2
            count += 1

    std_deviation = math.sqrt(sum / count)

    # Calculate average valuse
    # In this case, we remove outliers
    avg_x = 0; avg_y = 0; avg_t = 0
    avg_count = 0
    for xx, yy, tt in zip(x_dist, y_dist, t_dist):
        if math.fabs(tt - mean_t) <= std_deviation and tt != 0:
            avg_x += xx; avg_y += yy; avg_t += tt;
            avg_count += 1

    if avg_count > 0:
        avg_x /= avg_count; avg_y /= avg_count; avg_t /= avg_count;

    return avg_x, avg_y, avg_t


print(gstreamer_pipeline(flip_method=0))
# video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
# video_capture = cv2.VideoCapture('avinash 8_97.avi')
# video_capture = cv2.VideoCapture('Screencast 2022-05-21 00_09_26.avi')
# video_capture = cv2.VideoCapture('Screencast 2022-05-21 00_08_37.avi')
video_capture = cv2.VideoCapture('8_seconds.mp4')
# video_capture = cv2.VideoCapture('9_seconds.mp4')

# Create params for ShiTomasi corner detection
shitomasi_feature_params = dict(maxCorners=100, qualityLevel=0.6, minDistance=10, blockSize=10)

# Create params for lucas kanade optical flow
lk_params = dict(winSize=(14, 14),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

# Create some(100) random colors-------100 rows and 3 colours rgb
color = np.random.randint(0, 255, (100, 3))

frame_idx = 0
detect_interval = 5  # Till this number of frame FP will be traced and after this again new FP will generated and tracked

if video_capture.isOpened():

    # Properties of captured video
    # frames_count, fps, width, height = video_capture.get(cv2.CAP_PROP_FRAME_COUNT), video_capture.get(cv2.CAP_PROP_FPS), video_capture.get(
    # cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(frames_count, fps, width, height)

    # Testing
    average_distance_array = np.arange(0)
    average_velocity_array = np.arange(0)

    p0 = np.array([])
    p1 = np.array([])

    while True:

        # Capture a frame
        ok, frame = video_capture.read()

        if not ok:
            print('No frames grabbed!')
            break

        # Convert it to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                                   **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) != 0 and len(good_old) != 0:

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = (int(x) for x in new.ravel())
                    c, d = (int(x) for x in old.ravel())
                    # print(a,b,c,d)
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    cv2.circle(frame, (a, b), 10, color[i].tolist(), -1)

                frame_x_dist = np.zeros_like(frame_x_dist)
                frame_y_dist = np.zeros_like(frame_y_dist)
                frame_t_dist = np.zeros_like(frame_y_dist)
                index_array_new = np.zeros(len(good_new))
                for i, pt_s in enumerate(good_new):
                    xx1, yy1 = (x for x in pt_s.ravel())
                    rx1, ry1 = convert_screen_to_real(xx1, yy1)
                    min_index = 0
                    min_val = 999
                    x_dist = 0
                    y_dist = 0
                    for j, pt_e in enumerate(good_old):
                        xx2, yy2 = (x for x in pt_e.ravel())
                        rx2, ry2 = convert_screen_to_real(xx2, yy2)
                        dist = math.sqrt((rx1 - rx2) ** 2 + (ry1 - ry2) ** 2)
                        if min_val > dist:
                            min_val = dist
                            min_index = j
                            x_dist = math.fabs(rx1 - rx2)
                            y_dist = math.fabs(ry1 - ry2)
                    index = index_array_old[min_index]
                    index_array_new[i] = index
                    frame_x_dist[int(index)] = x_dist
                    frame_y_dist[int(index)] = y_dist
                    frame_t_dist[int(index)] = min_val

                frame_average_x_dist, frame_average_y_dist, frame_average_distance =\
                    Calculate_mean_by_distribution(frame_x_dist, frame_y_dist, frame_t_dist)

                frame_average_x_velocity = frame_average_x_dist * 30
                frame_average_y_velocity = frame_average_y_dist * 30
                frame_average_velocity = frame_average_distance * 30

                print("vx: " + str(frame_average_x_velocity))
                print("vy: " + str(frame_average_y_velocity))
                print("frame_velocity: " + str(frame_average_velocity))
                print()

                # Test data calculation
                average_distance_array = np.append(average_distance_array, frame_average_distance)
                average_velocity_array = np.append(average_velocity_array, frame_average_velocity)

        if p0 is None or len(p0) == 0 or \
            good_new is None or len(good_new) == 0 or \
            good_old is None or len(good_old) == 0 or \
            frame_idx % detect_interval == 0:

            # Getting vector of coordinates of feature point to be detected based on given shitomasi feature params
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **shitomasi_feature_params)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(frame)

            # creating evenly spaced index array of length equal to that of p0
            index_array_old = np.arange(len(p0))
            # print(index_array_old)

            # creating frame distance array length p0 and initialize with zeros
            frame_x_dist = np.zeros(len(p0))
            frame_y_dist = np.zeros(len(p0))
            frame_t_dist = np.zeros(len(p0))

        cv2.imshow('Optical flow sparse', mask)
        cv2.imshow('Frame', frame)

        # Force display, quitting on ESC
        if (cv2.waitKey(30) & 0xff) == 27:
            break

        old_gray = frame_gray.copy()
        frame_idx += 1

    real_avg_velocity = average_velocity_array.mean()
    print("Total_avg_velocity: " + str(real_avg_velocity))

else:
    print("Error: Unable to open camera")
