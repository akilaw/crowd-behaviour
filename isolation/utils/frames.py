import cv2
import pandas as pd
import os

def save_images(video_path, annotations_path, path, drawBoundindBox=False):

  x_ratio = 1/3840
  y_ratio = 1/2160

  if not os.path.exists(path):
    os.mkdir(path)

  for filename in os.listdir(video_path):
    print(filename)
    video = os.path.splitext(filename)[0]

    annotations = pd.read_csv(annotations_path + video + '.txt', delimiter=' ', header=None)

    vidcap = cv2.VideoCapture(video_path + filename)
    success,image = vidcap.read()
    frame = 0

    while success:
      output = cv2.resize(image, (640, 640), interpolation = cv2.INTER_AREA)
      cv2.imwrite("%s/%s_%d.jpg" % (path, video, frame), output)     # save frame as JPEG file      
      success, image = vidcap.read()

      if drawBoundindBox:
        selected = annotations.loc[annotations[5] == frame]
        for index, row in selected.iterrows():
          x1 =  round(row[1] * x_ratio)
          y1 =  round(row[2] * y_ratio)

          x2 =  round(row[3] * x_ratio)
          y2 =  round(row[4] * y_ratio)

          x_center = x_ratio * (row[1] + row[3]) / 2
          y_center = y_ratio * (row[2] + row[4]) / 2

          width = x_ratio * (row[3] - row[1])
          height = y_ratio * (row[4] - row[2])

          x1 =  round((x_center - width/2) * 640)
          y1 =  round((y_center - height/2) * 640)

          x2 =  round((x_center + width/2) * 640)
          y2 =  round((y_center + height/2) * 640)

          cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite("%s/%s_%d_b.jpg" % (path, video, frame), output) 

      frame += 1
      if frame > 10 and drawBoundindBox:
        break

save_images('../data/train/Drone1/Noon/', '../data/train/annotations/', '../data/test/images', False)
save_images('../data/train/Drone2/Noon/', '../data/train/annotations/', '../data/test/images', False)
