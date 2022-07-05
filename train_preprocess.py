import torch
from tqdm import tqdm
from config import cfg
from base import Trainer

def main():
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_loader()
    print("end")

if __name__ == '__main__':
    main()



