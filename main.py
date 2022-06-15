from models.FCN import *
from ModelHandler import *

if __name__ == "__main__":
    handler = ModelHandler("config")
    handler.train() ## Returns Train loss, Validation Loss, Val IOU Score, Val Pixel Accuracy

    IoU, pixel_acc = handler.test()
    print(IoU, pixel_acc)