"# PWADL_SoSe26"
Neue powershell Befehle:
    Split erzeugen / überschreiben:
    python -c "from src.data import create_split_csv; create_split_csv(force=True)"

    Bestehenden Split beibehalten:
    python -c "from src.data import create_split_csv; create_split_csv(force=False)"

    Split-Datei sichern:
    copy data\splits.csv data\splits_final.csv

    Testsplit-Klassenverteilung anzeigen:
    python -c "from src.data import get_metadata_for_split; df=get_metadata_for_split('test'); print(len(df)); print(df['yawning'].value_counts())"

    Trainval-Klassenverteilung anzeigen:
    python -c "from src.data import get_metadata_for_split; df=get_metadata_for_split('trainval'); print(len(df)); print(df['yawning'].value_counts())"

    Prüfen, ob IDs zwischen Splits überlappen:
    python -c "from src.data import load_split_csv; df=load_split_csv(); train=set(df[df.split=='train'].id.astype(str)); val=set(df[df.split=='val'].id.astype(str)); test=set(df[df.split=='test'].id.astype(str)); print('train-val', len(train&val)); print('train-test', len(train&test)); print('val-test', len(val&test))"
    --> Sollte alles 0 sein

    Dataset ohne Augmentation testen:
    python -c "from src.data import YawDDDataset; ds=YawDDDataset('train', num_frames=8, train=False); x,y=ds[0]; print(x.shape, y)"

    Dataset mit Augmentation testen:
    python -c "from src.data import YawDDDataset; ds=YawDDDataset('train', num_frames=8, train=True); x,y=ds[0]; print(x.shape, y)"

    Mini Training:
    python main.py --num_frames 8 --epochs 1 --n_trials 1 --n_splits_cv 3

    Training mit anderer Patience:
    python main.py --num_frames 32 --epochs 30 --n_trials 10 --n_splits_cv 5 --patience 7

    Checkpoint grob inspizieren:
    python -c "import torch; ckpt=torch.load('best_model_final.pt', map_location='cpu'); print(ckpt.keys() if isinstance(ckpt, dict) else 'state_dict only')"
    --> erwartet: dict_keys(['model_state', 'dropout', 'threshold', 'freeze_backbone', 'batch_size', 'lr', 'num_frames', 'cv_best_f1', 'final_train_f1'])

    Tensorboard starten:
    tensorboard --logdir runs
    http://localhost:6006

    Ausgabe eines Trainings in Datei speichern:
    python main.py --num_frames 32 --epochs 20 --n_trials 10 --n_splits_cv 5 *> training_log.txt
    oder
    python main.py --num_frames 32 --epochs 20 --n_trials 10 --n_splits_cv 5 2>&1 | Tee-Object -FilePath training_log.txt

    Tester-Ausgabe in Datei speichern:
    python Tester_Proto.py *> tester_log.txt
    oder:
    python Tester_Proto.py 2>&1 | Tee-Object -FilePath tester_log.txt
