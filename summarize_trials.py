import os
import glob
import re
import argparse
import numpy as np

from tensorboard.backend.event_processing import event_accumulator



"""
IM REGELFALL NICHT VERWENDEN:
Dieses Skript kann verwendet werden, wenn das finale Training nicht korrekt durchgelaufen ist und 
somit wiederholt werden muss. Die vorhandenen Tensorboard-Daten werden nach dem besten Trail durchsucht, 
damit diese Hyperparameter verwendet werden können
"""


def extract_trial_number(path):
    """
    Extrahiert die Trial-Nummer aus einem Pfad wie runs/optuna_trial_3.
    """

    match = re.search(r"optuna_trial_(\d+)", path)

    if match is None:
        return 999999

    return int(match.group(1))


def extract_fold_number(path):
    """
    Extrahiert die Fold-Nummer aus einem Pfad wie fold_2.
    """

    match = re.search(r"fold_(\d+)", path)

    if match is None:
        return 999999

    return int(match.group(1))


def get_scalar_values_from_dir(log_dir, tag):
    """
    Liest alle Werte eines Scalar-Tags aus einem TensorBoard-Logordner.
    """

    values = []

    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={"scalars": 0}
        )

        ea.Reload()

        scalar_tags = ea.Tags().get("scalars", [])

        if tag not in scalar_tags:
            return values

        events = ea.Scalars(tag)

        values = [event.value for event in events]

    except Exception as e:
        print(f"Warnung: Konnte {log_dir} nicht lesen: {e}")

    return values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--expected_folds", type=int, default=5)
    parser.add_argument("--tag", type=str, default="val/F1")

    args = parser.parse_args()

    trial_dirs = sorted(
        glob.glob(os.path.join(args.runs_dir, "optuna_trial_*")),
        key=extract_trial_number
    )

    if len(trial_dirs) == 0:
        print("Keine optuna_trial_* Ordner gefunden.")
        return

    results = []

    for trial_dir in trial_dirs:
        trial_number = extract_trial_number(trial_dir)

        fold_dirs = sorted(
            glob.glob(os.path.join(trial_dir, "fold_*")),
            key=extract_fold_number
        )

        fold_best_f1s = []

        print(f"\nTrial {trial_number}")

        for fold_dir in fold_dirs:
            fold_number = extract_fold_number(fold_dir)

            values = get_scalar_values_from_dir(fold_dir, args.tag)

            if len(values) == 0:
                print(f"  Fold {fold_number}: kein {args.tag} gefunden")
                continue

            best_f1 = max(values)
            best_epoch = int(np.argmax(values)) + 1

            fold_best_f1s.append(best_f1)

            print(
                f"  Fold {fold_number}: "
                f"best {args.tag} = {best_f1:.4f} "
                f"in Epoche {best_epoch}"
            )

        if len(fold_best_f1s) > 0:
            mean_f1 = float(np.mean(fold_best_f1s))
            std_f1 = float(np.std(fold_best_f1s))
        else:
            mean_f1 = float("nan")
            std_f1 = float("nan")

        complete = len(fold_best_f1s) == args.expected_folds

        print(
            f"  Ergebnis: mean={mean_f1:.4f}, std={std_f1:.4f}, "
            f"folds={len(fold_best_f1s)}/{args.expected_folds}, "
            f"{'VOLLSTÄNDIG' if complete else 'UNVOLLSTÄNDIG'}"
        )

        results.append(
            {
                "trial": trial_number,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "num_folds": len(fold_best_f1s),
                "complete": complete,
            }
        )

    complete_results = [r for r in results if r["complete"]]

    print("\n============================================================")
    print("Zusammenfassung vollständiger Trials")
    print("============================================================")

    if len(complete_results) == 0:
        print("Keine vollständigen Trials gefunden.")
        print("Du kannst den besten unvollständigen Trial nur mit Vorsicht verwenden.")
        return

    complete_results = sorted(
        complete_results,
        key=lambda x: x["mean_f1"],
        reverse=True
    )

    for r in complete_results:
        print(
            f"Trial {r['trial']}: "
            f"cv_mean_val_f1={r['mean_f1']:.4f}, "
            f"cv_std_val_f1={r['std_f1']:.4f}"
        )

    best = complete_results[0]

    print("\n============================================================")
    print("Bester vollständiger Trial")
    print("============================================================")
    print(f"Trial ID       : {best['trial']}")
    print(f"cv_mean_val_f1 : {best['mean_f1']:.4f}")
    print(f"cv_std_val_f1  : {best['std_f1']:.4f}")


if __name__ == "__main__":
    main()