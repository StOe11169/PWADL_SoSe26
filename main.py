import argparse, time
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from src.utils import setup_env
from src.data import YawDDDataset
from src.training import trainer, YawDDclassifier
from src.evaluation import evaluate


def main():

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='YawDD')
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # set seed and precision and get device
    setup_env(seed=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data preparation
    trainset = YawDDDataset('train', steps=args.steps)
    valset = YawDDDataset('val', steps=args.steps)
    testset = YawDDDataset('test', steps=args.steps)

    # dataloader 
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # model
    model = YawDDclassifier().to(device)
    summary(model)
    
    # start training
    trainer(trainloader=trainloader,
            valloader=valloader,
            model=model,
            epochs=2,
            lr=0.001,
            device=device
            )
    
    # test
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    test_metrics = evaluate(testloader, model, device)
    print(f"=================================================================\nTest Acc: {test_metrics['accuracy']:.3f}") 


if __name__ == "__main__":
    # get start time
    start_timestamp = time.time()

    # train model
    main()

    # info on training time
    time_passed = time.time()-start_timestamp
    print(f'\nTraining finished in {time_passed//3600}h {(time_passed%3600)//60}min {time_passed%60:.0f}s\n')