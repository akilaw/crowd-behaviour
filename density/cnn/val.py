import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
from models import vgg as SFANet
import argparse
from PIL import Image
import cv2
import sys

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


sys.path.insert(0, 'cnn2')

from matplotlib import pyplot as plt
from matplotlib import cm as CM

datasets = Crowd(os.path.join('../data/UCF-QNRF_ECCV18_PROCESSED', 'test'), 256, 8, is_gray=False, method='val')
dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                         num_workers=8, pin_memory=False)
model = SFANet.Model()
device = torch.device('cuda')
model.to(device)
model.load_state_dict(torch.load(os.path.join('./output', 'best_model.pth'), device))
model.eval()

epoch_minus = []
preds = []
gts = []

met = []
for i in range(len(preds)):
    met.append(100 * np.abs(preds[i] - gts[i]) / gts[i])

idxs = []
for i in range(len(met)):
    idxs.append(np.argmin(met))
    if len(idxs) == 5: break
    met[np.argmin(met)] += 100000000
    
set(idxs)


def resize(density_map, image):
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    image= image[0]
    #print(density_map.shape)
    result_img = np.zeros((density_map.shape[0]*2, density_map.shape[1]*2))
    for i in range(result_img.shape[0]):
        for j in range(result_img.shape[1]):
            result_img[i][j] = density_map[int(i / 2)][int(j / 2)] / 4
    result_img  = result_img.astype(np.uint8, copy=False)
    return result_img

def vis_densitymap(o, den, cc, img_path):
    fig=plt.figure()
    columns = 2
    rows = 1
#     X = np.transpose(o, (1, 2, 0))
    X = o
    summ = int(np.sum(den))
    
    den = resize(den, o)
    
    for i in range(1, columns*rows +1):
        # image plot
        if i == 1:
            img = X
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.imshow(img)
            
        # Density plot
        if i == 2:
            img = den
            fig.add_subplot(rows, columns, i)
            plt.gca().set_axis_off()
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.text(1, 80, 'SFANet* Est: '+str(summ)+', Gt:'+str(cc), fontsize=7, weight="bold", color = 'w')
            plt.imshow(img, cmap=CM.jet)
    
    filename = img_path.split('/')[-1]
    filename = filename.replace('.jpg', '_heatpmap.png')
    print('Save at', filename)
    plt.savefig('viz/'+filename, transparent=True, bbox_inches='tight', pad_inches=0.0, dpi=200)
    return (cc, summ)

processed_dir = '../data/UCF-QNRF_ECCV18_PROCESSED/test'
model.eval()
c = 0

actual = []
predicted = []

for inputs, count, name in dataloader:
    img_path = os.path.join(processed_dir, name[0]) + '.jpg'
    if c < 20:#c in set(idxs):
        inputs = inputs.to(device)
        torch.set_grad_enabled(False)
        outputs = model(inputs)
        
        img = Image.open(img_path).convert('RGB')
        height, width = img.size[1], img.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        img = cv2.resize(np.array(img), (width,height), cv2.INTER_CUBIC)
        
        act, pred = vis_densitymap(img, outputs.cpu().detach().numpy(), int(count.item()), img_path)
        print(act, pred)
        actual.append(act)
        predicted.append(pred)
        c += 1
            
    else:
        c += 1

mse = mean_squared_error(actual, predicted, squared=False)
mae = mean_absolute_error(actual, predicted)
print("MSE : " + str(mse) + "MAE : " + str(mae))