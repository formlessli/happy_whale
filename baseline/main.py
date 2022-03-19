from train import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, choices=["resnet50, resnet101, vgg16"], default="resnet50",
                    help="model name")
parser.add_argument("--root-path", type=str, default="CACD2000/",
                    help="root path of images")
parser.add_argument("--num-classes", type=int, default=30,
                    help="number of classes")
parser.add_argument("--model-path", type=str, default="./model/params.pkl",
                    help="path to save and load model")
parser.add_argument("--num-epoch", type=int, default=20,
                    help="number of epoch")
parser.add_argument("--batch-size", type=int, default=64,
                    help="batch size")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--batch-display", type=int, default=50,
                    help="frequency of batch to display result")
parser.add_argument("--save-freq", type=int, default=1,
                    help="frequency to save model")
parser.add_argument("--pretrained", type=int, default=0,
                    help="Load pretrained model or not")
parser.add_argument("--data_path", type=str, default="/kaggle/input/happy-whale-and-dolphin",
                    help="root path of images")

args = parser.parse_args()




if __name__ == '__main__':
    train_whale = Train(data_path = args.data_path, model_name = args.model, number_classes = args.num_classes, path=args.model_path, loadPretrain=args.pretrained)
    
    train_whale.start_train(epoch=args.num_epoch, batch_size=args.batch_size,
                          learning_rate=args.lr, batch_display=args.batch_display, save_freq=args.save_freq)
