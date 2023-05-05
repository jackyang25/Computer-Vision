from Algorithm.main import *
import os
import matplotlib
import cv2
import numpy as np

matplotlib.use('TKAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow output


def det_move(obj_x_coord, obj_y_coord, xres, yres):
    centerx, centery = xres / 2, yres / 2
    move_x = obj_x_coord - centerx
    move_y = obj_y_coord - centery
    if move_x != 0:
        move_x /= abs(obj_x_coord - centerx)
    if move_y != 0:
        move_y /= abs(obj_y_coord - centery)
    return move_x, move_y


def main(_argv):
    model = get_model()  # load saved model

    while True:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            results = model(frame)
            dataframe = results.pandas().xyxy[0]
            isEmpty = dataframe.empty

            if not isEmpty:
                center = int(cap.get(4) / 2)
                xLeft = dataframe.iat[0, 0]
                xRight = dataframe.iat[0, 2]
                yLower = dataframe.iat[0, 1]
                yUpper = dataframe.iat[0, 3]
                confidence = dataframe.iat[0, 4]
                item = dataframe.iat[0, 6]
                group = dataframe.index[-1]

                xCoor = int((xRight + xLeft) / 2)
                yCoor = int((yUpper + yLower) / 2)

                print(f"[group: {group}] [item: {item}] "
                      f"[coord: {xCoor, yCoor}] [confidence: {confidence}]")

            # print('\n', results.pandas().xyxy[0])

            cv2.imshow('YOLO', np.squeeze(results.render()))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
