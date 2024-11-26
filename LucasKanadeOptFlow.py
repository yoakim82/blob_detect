import os.path
from datetime import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.ma.core import zeros_like

''' 
    inRange checks whether the given cordinates line in the given image limits
 cordinates, limits are tuples i.e., (X,Y) 
'''
def inRange( cordinates, limits):
	x,y = cordinates
	X_Limit, Y_Limit = limits
	return 0 <= x and x < X_Limit and 0 <= y and y < Y_Limit

'''
    opticalFlow calculates the displacements in X and Y directions i.e., (u,v)
    given two consecutive images varying with time
'''
def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):

    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)

    old_frame = old_frame / 255
    new_frame = new_frame / 255

    #Convolve to get gradients w.r.to X, Y and T dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)              #Gradient over X
    fy = cv2.filter2D(old_frame, -1, kernel_y)              #Gradient over Y
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)  #Gradient over Time


    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature in feature_list:        #   for every corner
            j, i = feature.ravel()		#   get cordinates of the corners (i,j). They are stored in the order j, i
            i, j = int(i), int(j)		#   i,j are floats initially

            I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0],1))
            A = np.vstack((I_x, I_y)).T

            U = np.matmul(np.linalg.pinv(A), b)     # Solving for (u,v) i.e., U

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)


'''
Draw the displacement vectors on the image, given (u,v) and save it to the output filepath provided
'''
def drawOnFrame(frame, U, V, output_file):

    line_color = (0, 255, 0) #  Green

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            u, v = U[i][j], V[i][j]

            if u and v:
                frame = cv2.arrowedLine( frame, (i, j), (int(round(i+u)), int(round(j+v))),
                                        (0, 255, 0),
                                        thickness=1
                                    )
    cv2.imwrite(output_file, frame)


'''
Create a plot of the displacement vectors given (u,v) and plot the two images and displacement in a row.
Save the plot to given output filepath
'''
def drawSeperately(old_frame, new_frame, U, V, output_file):

    displacement = np.ones_like(img2)
    displacement.fill(255.)             #Fill the displacement plot with White background
    line_color =  (0, 0, 0)
    # draw the displacement vectors
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):

            start_pixel = (i,j)
            end_pixel = ( int(i+U[i][j]), int(j+V[i][j]) )

            #check if there is displacement for the corner and endpoint is in range
            if U[i][j] and V[i][j] and inRange( end_pixel, img1.shape ):     
                displacement = cv2.arrowedLine( displacement, start_pixel, end_pixel, line_color, thickness =2)

    figure, axes = plt.subplots(1,3)
    axes[0].imshow(old_frame, cmap = "gray")
    axes[0].set_title("first image")
    axes[1].imshow(new_frame, cmap = "gray")
    axes[1].set_title("second image")
    axes[2].imshow(displacement, cmap = "gray")
    axes[2].set_title("displacements")
    figure.tight_layout()
    plt.savefig(output_file, bbox_inches = "tight", dpi = 200)


def testcode():
    #   Read Input
    img1 = cv2.imread("./Inputs/grove1.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread("./Inputs/grove2.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Obtain (u,v) from Lucas Kanade's optical flow approach
    dx, dy = optical_flow( img1, img2, 3, 0.05)



    # Save results
    img2 = cv2.cvtColor( img2, cv2.COLOR_GRAY2RGB)
    drawSeperately(img1, img2, U, V, "./Results/Grove_Seperate_Result.png")
    drawOnFrame(img2, U, V, './Results/Grove_Result.png')


def draw_flow(img, flow, step=16):
    # from the beginning to position 2 (excluded channel info at position 3)
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    (h, w) = flow.shape[:2]
    (fx, fy) = (flow[:, :, 0], flow[:, :, 1])
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 0xFF
    hsv[..., 2] = np.minimum(v * 4, 0xFF)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow('hsv', bgr)
    return bgr





def warp_flow(img, flow):
    (h, w) = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def opt_flow_camera(pos, fnames, idx):

    useCamera = False

    if useCamera:
        cam = cv2.VideoCapture(0)
    else:

        cam = cv2.VideoCapture(os.path.join('Inputs', fnames[idx]))
        nof = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        pos_frame = int(pos * nof)
        cam.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
    (ret, prev) = cam.read()
    prev = cv2.resize(prev, (640, 480))

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = prev.copy()
    cv2.waitKey(50)

    useLukasKanade = False
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    flow_sample = cv2.calcOpticalFlowFarneback(prevgray, prevgray, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    if useCamera:
        img_cnt = 0
    else:
        img_cnt = pos_frame
    while True:
        (ret, img) = cam.read()
        img_small = cv2.resize(img, (640, 480))
        vis = img_small.copy()
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        img_cnt += 1
        if not useCamera:
            print(f"frame {img_cnt} ({int(100*img_cnt/nof)}%) processed.")

        if useLukasKanade:
            dx, dy = optical_flow(prevgray, gray, 150, 0.005)
            flow = np.dstack((dx, dy))

        else:

            flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None,0.5,5,50,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        flow *= 10
        #cv2.imshow('flow', draw_flow(gray, flow*50))

        if show_hsv:
            hsv = draw_hsv(flow)
            gray1 = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
            #hsv *= 50
            thresh = cv2.threshold(gray1, 15, 0xFF,
                                   cv2.THRESH_BINARY)[1]
            #thresh = cv2.dilate(thresh, None, iterations=0)
            #cv2.imshow('thresh', thresh)
            #cv2.waitKey(0)
            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for c in cnts:

                # if the contour is too small, ignore it
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 100 and h > 100 and w < 900 and h < 680:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0xFF, 0), 4)
                    cv2.putText(vis,"object",(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0xFF),1)
            #cv2.imshow('Image', vis)



        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv2.imshow('glitch', cur_glitch)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print ('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            useLukasKanade = not useLukasKanade
            print ('useLukasKanade is', ['off', 'on'][useLukasKanade])


        flow = draw_flow(gray, flow * 1)
        thresh_3d = np.stack((thresh,) * 3, axis=-1)
        gray1_3d = np.stack((gray1,) * 3, axis=-1)


        combin1 = np.hstack((flow, thresh_3d))
        combin2 = np.hstack((vis, hsv))
        combin = np.vstack((combin1, combin2))
        cv2.imshow('frame', combin)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    #opt_flow_camera(pos=0.05)
    fnames = ['0217.mov', 'drone_granso.mp4', '2024-05-28_11_13_20_visercam01_Cut_Long 1.mp4']
    opt_flow_camera(pos=0.1, fnames=fnames, idx=0)


