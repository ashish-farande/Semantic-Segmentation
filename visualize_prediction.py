import torch
import numpy as np
color2class = {
                #terrain
                (192,192,192): 0, (105,105,105): 0, (160, 82, 45):0, (244,164, 96): 0, \
                #vegatation
                ( 60,179,113): 1, (34,139, 34): 1, ( 154,205, 50): 1, ( 0,128,  0): 1, (0,100,  0):1, ( 0,250,154):1, (139, 69, 19): 1,\
                #construction
                (1, 51, 73):2, ( 190,153,153): 2, ( 0,132,111): 2,\
                #vehicle
                (0,  0,142):3, ( 0, 60,100):3, \
                #sky
                (135,206,250):4,\
                #object
                ( 128,  0,128): 5, (153,153,153):5, (255,255,  0 ):5, \
                #human
                (220, 20, 60):6, \
                #animal
                ( 255,182,193):7,\
                #void
                (220,220,220):8, \
                #undefined
                (0,  0,  0):9
        }

class2color = {v: k for k, v in color2class.items()}
class_names = ['terrain', 'vegatation', 'construction', 'vehicle', 'sky', 'object', 'human', 'animal', 'void', 'undefined']

class2name = {}
for i in range(len(class_names)):
    class2name[i]=class_names[i]

# This test example have 8 classes. So I picked it!
input_ = torch.Tensor(np.load('good_test_input.npy'))
label = torch.Tensor(np.load('good_test_label.npy'))

def prediction_to_color(pred):
    ''' make prediction to 3*H*W matrix to visualize'''
    prediction = pred.cpu().detach().numpy()[0,:,:]
    colors_ofimg = np.zeros((3, prediction.shape[0], prediction.shape[1])) # change to numpy att  h*w*3
    for i in range(prediction.shape[0]):
        
        for j in range(prediction.shape[1]):
            pixel = prediction[i,j]
            color=class2color[pixel]

            colors_ofimg[:, i,j]=color
    return colors_ofimg

def get_prediction(best_model):
    ''' given the best model, forward input and yield labels '''
    pred = torch.argmax(best_model.forward(input_), dim=1) 
    
    return pred
    
# generate img and labels
lbl_colors = prediction_to_color(label)
img = input_[0,:,:,:].cpu().detach().numpy()
        
# make some legend! beautiful!
from matplotlib.patches import Patch
import matplotlib.colors as mpcolor

legend_elements = [
                   Patch(facecolor=mpcolor.to_rgba(np.array(class2color[cls])/255),
                         label=class2name[cls]) 
    for cls in class2name]

