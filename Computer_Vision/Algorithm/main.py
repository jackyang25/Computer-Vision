import torch


# returns model
def get_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='./Algorithm/pt_files/bottle.pt')  # or YOLOv5m, YOLOv5l, YOLOv5x, custom
    return model


# returns coordinates
def get_coordinates(frame, model):
    results = model(frame)  # using the model each frame
    rows = results.pandas().xyxy[0]

    if len(rows) != 0:
        x_min, y_min, x_max, y_max = int(rows['xmin'][0]), int(rows['ymin'][0]), int(rows['xmax'][0]), int(
            rows['ymax'][0])

        return x_min, y_min, x_max, y_max
    return None
