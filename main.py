import argparse, time
import torch
from torch.utils.data import DataLoader
from src.utils import setup_env
from src.data import CustomDataset
from src.training import trainer

def main():

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data')
    args = parser.parse_args()

    # set seed and precision and get device
    setup_env(seed=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data preparation
    trainset = CustomDataset("Train")
    valset = CustomDataset("Val")

    # # dataloader 
    # trainloader = DataLoader(trainset, batch_size=, num_workers=0, shuffle=True)
    # valloader = DataLoader(valset, batch_size=, num_workers=0, shuffle=False)

    # # model
    # model = 
    # summary(model)
    
    # # start training
    # trainer(trainloader=trainloader,
    #         valloader=valloader,
    #         model=model,
    #         device=device
    #         )
    
    # # test
    # test(loader=)


if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # train model
    main()

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')