import cv2
import pandas as pd
import os

x_ratio = 1/3840
y_ratio = 1/2160

classes = ['Calling', 'Carrying', 'Drinking', 'Hand Shaking', 'Hugging', 'Lying', 'Pushing/Pulling', 
              'Reading', 'Running', 'Sitting', 'Standing', 'Walking']

classes_map = dict(zip(classes, list(range(0, len(classes)))))
print(classes_map)

def create_labels(annotation_path, path):
  if not os.path.exists(path):
    os.mkdir(path)

  for filename in os.listdir(annotation_path):
    print(filename)
    video = os.path.splitext(filename)[0]

    annotations = pd.read_csv(annotation_path + filename, delimiter=' ', header=None)
    #print(annotations)

    annotations['x_center'] = x_ratio * (annotations[1] + annotations[3]) / 2
    annotations['y_center'] = y_ratio * (annotations[2] + annotations[4]) / 2

    annotations['width'] = x_ratio * (annotations[3] - annotations[1])
    annotations['height'] = y_ratio * (annotations[4] - annotations[2])

    annotations['class'] = annotations[10]
    annotations = annotations[[5, 'class', 'x_center', 'y_center', 'width', 'height']].dropna()

    annotations['class']  = annotations['class'].map(classes_map) 
    annotations_grouped = annotations.groupby(5)

    for group_name, df_group in annotations_grouped:
      output = df_group[['class', 'x_center', 'y_center', 'width', 'height']].dropna()
      output.to_csv(path + "/" + video + "_" +str(group_name) + ".txt", index=False, sep = ' ', header = False)

    '''
    annotations_grouped = annotations.groupby('class')
    for group_name, df_group in annotations_grouped:
      print(group_name)
    '''

create_labels('../data/train/annotations/', '../data/test/labels')


