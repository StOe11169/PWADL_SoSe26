import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import time
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import setup_env
from src.data import YawDDDataset
from src.training import trainer, YawDDclassifier
from src.evaluation import evaluate

"""
IM REGELFALL NICHT VERWENDEN:
Dieses Skript kann verwendet werden, um das finale Training mit den besten Hyperparametern manuell zu starten,
falls ein langer Trainingslauf fehlerhaft abgebrochen und somit nicht korrekt gespeichert wurde
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--freeze_backbone", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cv_best_f1", type=float, default=float("nan"))

    args = parser.parse_args()

    setup_env(seed=0)

    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Verwendetes Gerät:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    final_train_dataset_aug = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=True
    )

    final_train_dataset_eval = YawDDDataset(
        "trainval",
        num_frames=args.num_frames,
        train=False
    )

    final_trainloader = DataLoader(
        final_train_dataset_aug,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    final_train_eval_loader = DataLoader(
        final_train_dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = YawDDclassifier(args.dropout).to(device)

    final_state_path = "checkpoints/final_state_dict.pt"
    final_log_dir = f"runs/final_train_only_{time.strftime('%Y%m%d_%H%M%S')}"

    final_train_f1, final_epoch = trainer(
        trainloader=final_trainloader,
        valloader=None,
        train_eval_loader=final_train_eval_loader,
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        device=device,
        threshold=args.threshold,
        patience=args.patience,
        log_dir=final_log_dir,
        save_path=final_state_path,
        early_stopping=False
    )

    model.load_state_dict(torch.load(final_state_path, map_location=device))

    final_checkpoint_path = "best_model_final.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "dropout": args.dropout,
            "threshold": args.threshold,
            "freeze_backbone": args.freeze_backbone,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_frames": args.num_frames,
            "cv_best_f1": args.cv_best_f1,
            "final_train_f1": final_train_f1,
        },
        final_checkpoint_path
    )

    print(f"\nFinales Modell gespeichert unter: {final_checkpoint_path}")

    testset = YawDDDataset(
        "test",
        num_frames=args.num_frames,
        train=False
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_metrics = evaluate(
        testloader,
        model,
        device,
        args.threshold,
        writer=None,
        epoch=None,
        prefix="test",
        verbose=True
    )

    writer = SummaryWriter(final_log_dir)

    writer.add_hparams(
        hparam_dict={
            "batch_size": args.batch_size,
            "freeze_backbone": args.freeze_backbone,
            "lr": args.lr,
            "dropout": args.dropout,
            "threshold": args.threshold,
            "num_frames": args.num_frames,
        },
        metric_dict={
            "cv_best_f1": args.cv_best_f1,
            "final_train_f1": final_train_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        },
        run_name="final_train_only"
    )

    writer.close()

    print("\n================ FINAL TEST ================")
    print(f"Test Accuracy : {test_metrics['accuracy']:.3f}")
    print(f"Test Precision: {test_metrics['precision']:.3f}")
    print(f"Test Recall   : {test_metrics['recall']:.3f}")
    print(f"Test F1       : {test_metrics['f1']:.3f}")

    if test_metrics["roc_auc"] == test_metrics["roc_auc"]:
        print(f"Test ROC-AUC  : {test_metrics['roc_auc']:.3f}")
    else:
        print("Test ROC-AUC  : nan")

    if test_metrics["pr_auc"] == test_metrics["pr_auc"]:
        print(f"Test PR-AUC   : {test_metrics['pr_auc']:.3f}")
    else:
        print("Test PR-AUC   : nan")

    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()