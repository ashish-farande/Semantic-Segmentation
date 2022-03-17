from basic_fcn import *
from dataloader import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy
import configparser
import sys
import pandas as pd

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

def train(outdir):
    ''' train for many epochs, save best model, save train/val performance '''
    
    # before train evaluate first
    miou_score, acc, loss, class_iou = evaluate(fcn_model, val_loader)
    
    best_iou_score = 0.0
    
    train_losses = [np.nan]
    val_loss = [loss]
    val_miou = [miou_score]
    val_acc = [acc]
    val_class_iou = [class_iou]
    
    for epoch in range(epochs):
        ts = time.time()
        train_loss_in_epoch = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's # batch_size, 3, 384, 768
            labels = labels.to(device)#transfer the labels to the same device as the model's # batch_size, 384, 768

            outputs = fcn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            loss = criterion(outputs, labels.long()) #calculate loss
            
            train_loss_in_epoch.append(loss.item())
            
            # backpropagate
            loss.backward()
            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        mean_train_loss = np.mean(train_loss_in_epoch)
        current_miou_score, mean_val_acc, mean_val_loss, class_iou = evaluate(fcn_model, val_loader)
        
        val_loss.append(mean_val_loss)
        val_miou.append(current_miou_score)
        val_acc.append(mean_val_acc)
        val_class_iou.append(class_iou)
        train_losses.append(mean_train_loss)
        
        print(f'==== Epoch {epoch} ======')
        print(f"Train Loss is {mean_train_loss}")
        print(f"Val Loss is {mean_val_loss}")
        print(f"IoU is {current_miou_score}")
        print(f"Pixel acc is {mean_val_acc}")
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            torch.save(fcn_model.state_dict(), os.path.join(outdir,'best-model.pt'))
        
        
        # save training curves every epoch. If need to terminate early should be fine too
        perf = pd.DataFrame([train_losses, val_loss, val_miou, val_acc], 
                     index = ['train_loss', 'val_loss', 'val_avg_miou', 'val_acc'])
        class_iou_df = pd.DataFrame(val_class_iou)

        perf.to_csv(os.path.join(outdir,'perf.csv'))
        class_iou_df.to_csv(os.path.join(outdir,'iou.csv'))

    

    
def evaluate(model, dataset_loader):
    '''
    Calculate the validation loss, IoU and pixel acc at
    '''
    model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []
    class_specific_iou = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input_, label) in enumerate(dataset_loader):

            # both inputs and labels have to reside in the same device as the model's
            input_ = input_.to(device) #transfer the input to the same device as the model's
            label = label.to(device) #transfer the labels to the same device as the model's

            output = model(input_)

            loss = criterion(output, label.long()) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(model.forward(input_), dim=1) # Make sure to include an argmax to get the prediction from the outputs of your model ##### NOT SURE IF DIMESION IS RIGHT
            
            iou_for_each_class = iou(pred, label, n_class)
            class_specific_iou.append(iou_for_each_class)
            mean_iou_scores.append(np.nanmean(iou_for_each_class))  # Complete this function in the util, notice the use of np.nanmean() here
            
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util

    
    mean_loss = np.mean(losses)
    mean_iou_scores_all = np.mean(mean_iou_scores)
    mean_pixel_acc = np.mean(accuracy)
    class_iou = np.nanmean(np.stack(class_specific_iou), axis = 0)
    

    model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return mean_iou_scores_all, mean_pixel_acc, mean_loss, class_iou

def test(outdir):
    #TODO: load the best model and complete the rest of the function for testing
    best_model = FCN(n_class = n_class).to(device)
    best_model.load_state_dict(torch.load(os.path.join(outdir,'best-model.pt')))
    
    miou_score, mean_test_acc, mean_test_loss, class_iou = evaluate(best_model, test_loader)
    
    return miou_score, mean_test_acc, mean_test_loss, class_iou
    
if __name__ == "__main__":
    
    # read config
    config_file=sys.argv[1]
    cps = configparser.ConfigParser()
    cps.read(config_file)
    
    # make output directory
    outdir=cps['Files']['outdir']
    try:
        os.mkdir(outdir)
    except Exception as e:
        print(e)
    
    
    # set parameters
    batch_size = int(cps['Training params']['batch_size'])
    lr=float(cps['Training params']['lr'])
    epochs = int(cps['Training params']['epoch'])
    

    # read dataset
    crop = False if cps['Data augument']['crop']=='False' else True
    hf = False if cps['Data augument']['horizontal_flip']=='False' else True
    vf = False if cps['Data augument']['vertical_flip']=='False' else True
    jitter = False if cps['Data augument']['color_jitter']=='False' else True
    
    
    train_dataset = TASDataset('tas500v1.1', crop = crop, 
                               horizontal_flip = hf,
                              vertical_flip = vf,
                              color_jitter = jitter) 
    val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
    test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=False)

    
    # Load model and loss
    criterion = nn.CrossEntropyLoss() # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
    n_class = 10
    fcn_model = FCN(n_class=n_class)
    fcn_model.apply(init_weights)
    optimizer = optim.Adam(fcn_model.parameters(), lr=lr) # choose an optimizer

    # Device!
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # determine which device to use (gpu or cpu)
    fcn_model = fcn_model.to(device) #transfer the model to the device
    
    # Do it
    
    
    
    train(outdir = outdir)
    
    # test
    test_miou_score, mean_test_acc, mean_test_loss, class_iou = test(outdir)
    pd.DataFrame([[test_miou_score, mean_test_acc, mean_test_loss]+list(class_iou)]).to_csv(
        os.path.join(outdir,'test_perf.csv')
    )
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()