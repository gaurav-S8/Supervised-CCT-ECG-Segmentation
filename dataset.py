import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def gatherSplit(df, patList, fs, setType):
    """
    Train:
        For each record, extract two fixed 4s windows: [2–6)s and [4–8)s  (window_len = 4 * fs)
    Val/Test:
        For each record, keep the continuous 8s span [1–9)s  (window_len = 8*fs),
        i.e., drop the first 1s and last 1s.

    Returns:
        X: Tensor [N, 1, L]
        y: Tensor [N, L] (long)
    """
    
    st = setType.lower()
    if st == "train":
        win_len = 4 * fs
        startA  = 2 * fs
        startB  = 4 * fs
        # Must have at least 8s to take [2–6) and [4–8)
        need_len = 8 * fs
    elif st in ("val", "test"):
        # 8 seconds
        win_len = 8 * fs
        # Begin at 1s
        startA  = 1 * fs
        # Must reach exactly 9s index
        need_len = startA + win_len
    else:
        raise ValueError("setType must be 'train', 'val'/'validation', or 'test'")

    X, y = [], []
    subDf = (
        df[df["record_number"].isin(patList)]
        .sort_values(["record_number", "lead_index"])
    )

    for _, row in subDf.iterrows():
        sig = row["signal"]
        lab = row["label_wave"]
        if len(sig) < need_len:
            continue
        # Training Set
        if st == "train":
            # 2–6 s
            X.append(sig[startA : startA + win_len])
            y.append(lab[startA : startA + win_len])
            # 4–8 s
            X.append(sig[startB : startB + win_len])
            y.append(lab[startB : startB + win_len])
        # Val/Test set: 1–9 s
        else:
            X.append(sig[startA : startA + win_len])
            y.append(lab[startA : startA + win_len])

    if not X:
        # Return empty tensors with the right lengths for the split
        print("HERE")
        return (
            torch.empty((0, win_len), dtype = torch.float32).unsqueeze(1),
            torch.empty((0, win_len), dtype = torch.long),
        )

    return (
        # [N, 1, L]
        torch.tensor(np.stack(X), dtype = torch.float32).unsqueeze(1),
        # [N, L]
        torch.tensor(np.stack(y), dtype = torch.long),      
    )

def buildLoaders_1Lead(df, fs, split = (0.7, 0.15, 0.15)):
    """
    Setup 1: PATIENT-WISE HOLDOUT (approx. 70/15/15 split).
    
    - Patients are sorted by their {record_number} and divided into train/val/test groups 
      according to the provided fractions.
    - For each split, gatherSplit() generates fixed-length windows (e.g. 2–6s, 4–8s) 
      and returns corresponding tensors.
    - This split allows overlap of patients' signal windows across sets 
      if patient IDs are not carefully controlled in gatherSplit, 
      but with the current patient-based division there is no overlap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a 'record_number' column indicating patient IDs.
    fs : int, default=500
        Sampling frequency, passed to gatherSplit.
    split : tuple of float, default=(0.7, 0.15, 0.15)
        Fractions for train/val/test. Must sum to 1.0.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test', 
        each containing {"X": Tensor, "y": Tensor}.
    """
    
    tr, va, te = split
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError(f"split must sum to 1.0, got {split}")

    patients = np.array(sorted(df["record_number"].unique()))
    if patients.size == 0:
        raise ValueError("No patients found in df['record_number'].")

    nTrain = int(tr * len(patients))
    nVal = int(va * len(patients))

    trainPat = patients[ : nTrain]
    valPat = patients[nTrain : nTrain + nVal]
    testPat = patients[nTrain + nVal : ]

    Xtrain, ytrain = gatherSplit(df, trainPat, fs, "train")
    Xval, yval = gatherSplit(df, valPat, fs, "val")
    Xtest, ytest  = gatherSplit(df, testPat, fs, "test")

    return {
        "train": {"X": Xtrain, "y": ytrain},
        "val":   {"X": Xval,   "y": yval},
        "test":  {"X": Xtest,  "y": ytest},
    }

def saveTensorSplits(splits, cacheDir, prefix, meta = None):
    """
    Save tensor splits to disk:
      {prefix}_train_X.pt, {prefix}_train_y.pt, ... plus {prefix}_meta.json (optional)
    """
    os.makedirs(cacheDir, exist_ok = True)

    torch.save(splits["train"]["X"], os.path.join(cacheDir, f"{prefix}_train_X.pt"))
    torch.save(splits["train"]["y"], os.path.join(cacheDir, f"{prefix}_train_y.pt"))
    torch.save(splits["val"]["X"],   os.path.join(cacheDir, f"{prefix}_val_X.pt"))
    torch.save(splits["val"]["y"],   os.path.join(cacheDir, f"{prefix}_val_y.pt"))
    torch.save(splits["test"]["X"],  os.path.join(cacheDir, f"{prefix}_test_X.pt"))
    torch.save(splits["test"]["y"],  os.path.join(cacheDir, f"{prefix}_test_y.pt"))

    meta = meta or {}
    meta["prefix"] = prefix
    meta["shapes"] = {
        "train_X": list(splits["train"]["X"].shape),
        "train_y": list(splits["train"]["y"].shape),
        "val_X":   list(splits["val"]["X"].shape),
        "val_y":   list(splits["val"]["y"].shape),
        "test_X":  list(splits["test"]["X"].shape),
        "test_y":  list(splits["test"]["y"].shape),
    }
    with open(os.path.join(cacheDir, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent = 2)

def makeSamples(df, fs, trimSeconds = 1.0, sortOn = "lead_index"):
    """
    Build patient-level samples from a dataframe of ECG signals, trimming by time.

    For each patient (grouped by 'record_number'):
      - Sorts leads by 'lead_index' (or 'channel' if not present).
      - Verifies all leads have identical length for that patient.
      - Removes the first and last `trim_seconds` of samples, based on `fs`.
      - Stacks signals into X with shape (C, T') and labels into L with shape (C, T').
      - Converts to PyTorch tensors (float32 for X, long for L).

    Parameters
    ----------
    df : pd.DataFrame
        Columns required:
          - 'record_number' (patient ID)
          - 'lead_index' or 'channel' (lead order)
          - 'signal' (1D array-like ECG)
          - 'label_wave' (1D array-like labels)
    fs : int
        Sampling rate in Hz (samples per second).
    trimSeconds : float, default=1.0
        Seconds to trim from BOTH the start and end of each lead.

    Returns
    -------
    list[tuple[int, torch.Tensor, torch.Tensor]]
        Each element: (recId, X, L)
          - recId : int
          - X : torch.FloatTensor, shape (C, T')
          - L : torch.LongTensor,  shape (C, T')
    """
    lead_key = "lead_index"
    samples = []
    
    if(sortOn == "channel"):
        lead_key = "channel"

    for recId, g in df.groupby("record_number"):
        g = g.sort_values(lead_key)

        signals, labels = [], []
        refLen = None
        lo = hi = None

        for _, row in g.iterrows():
            sig = np.asarray(row["signal"], dtype = np.float32)
            lab = np.asarray(row["label_wave"], dtype = np.uint8)

            if refLen is None:
                refLen = len(sig)
                cut = int(round(trimSeconds * fs))
                if refLen <= 2 * cut:
                    # too short to trim first/last `trimSeconds`
                    signals, labels = [], []
                    break
                lo, hi = cut, refLen - cut

            # Enforce identical lengths across leads for this patient
            if len(sig) != refLen or len(lab) != refLen:
                signals, labels = [], []
                break
                
            signals.append(sig[lo : hi])
            labels.append(lab[lo : hi])

        if not signals:
            continue

        # (C, T')
        X = torch.from_numpy(np.stack(signals, axis = 0))
        # (C, T')
        L = torch.from_numpy(np.stack(labels, axis = 0)).long()
        samples.append((int(recId), X, L))
    return samples


# Split samples into train/val/test (patient-wise, no shuffle)
def splitSamples(samples, split = (0.8, 0.1, 0.1)):
    """
    Patient-wise deterministic split with custom fractions.

    Parameters
    ----------
    samples : list[tuple[int, torch.Tensor, torch.Tensor]]
        (recId, X, L) per patient.
    split : (float, float, float)
        Fractions for (train, val, test). Must sum to 1.0.

    Returns
    -------
    (train_samples, val_samples, test_samples)
        Each is a list of (X, L) in original order (no shuffle).
    """
    tr, va, te = split
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError(f"split must sum to 1.0, got {split}")

    recIds = [rid for rid, _, _ in samples]
    uniqueRecs = sorted(set(recIds))  # deterministic
    n = len(uniqueRecs)

    # Contiguous counts (deterministic)
    nTrain = int(round(tr * n))
    nVal   = int(round(va * n))
    # Keep within bounds
    nTrain = min(nTrain, n)
    nVal   = min(nVal, n - nTrain)
    nTest  = n - nTrain - nVal

    trainRecs = set(uniqueRecs[: nTrain])
    valRecs   = set(uniqueRecs[nTrain : nTrain + nVal])
    testRecs  = set(uniqueRecs[nTrain + nVal : nTrain + nVal + nTest])

    def filterByRecs(recSet):
        # drop recId and keep (X, L) pairs, preserving original order
        return [(x, y) for rid, x, y in samples if rid in recSet]

    return (
        filterByRecs(trainRecs),
        filterByRecs(valRecs),
        filterByRecs(testRecs),
    )


def buildLoaders_12Leads(df, fs, trimSeconds, split: tuple[float, float, float] = (0.8, 0.1, 0.1), sortOn = "lead_index"):
    """
    Build patient-wise tensors for 12-lead ECG segmentation (no shuffling).

    Workflow
    --------
    1) makeSamples(df, fs, trimSeconds) -> [(recId, X, L), ...] where
       - X: FloatTensor (C, T'), L: LongTensor (C, T')
    2) splitSamples(samples, split) -> (train, val, test) lists of (X, L).
    3) Stack per split along batch dimension to return tensors:
         {
           "train": {"X": Xtrain, "y": ytrain},
           "val":   {"X": Xval,   "y": yval},
           "test":  {"X": Xtest,  "y": ytest},
         }
       Shapes:
         X*: (B, C, T'), y*: (B, C, T')

    Parameters
    ----------
    df : pd.DataFrame
        Must include 'record_number', 'lead_index' or 'channel', 'signal', 'label_wave'.
    fs : int
        Sampling rate (Hz).
    trimSeconds : float
        Seconds trimmed from both start and end of each lead.
    split : (float, float, float), default (0.8, 0.1, 0.1)
        Fractions for (train, val, test). Must sum to 1.0.

    Returns
    -------
    dict
        {
          "train": {"X": FloatTensor (B,C,T'), "y": LongTensor (B,C,T')},
          "val":   {"X": FloatTensor (B,C,T'), "y": LongTensor (B,C,T')},
          "test":  {"X": FloatTensor (B,C,T'), "y": LongTensor (B,C,T')},
        }
    """
    # Build samples
    samples = makeSamples(df, fs, trimSeconds, sortOn)
    assert len(samples) > 0, "No samples created from dataframe"

    # Split patient-wise (deterministic, no shuffle)
    trainS, valS, testS = splitSamples(samples, split)

    # Stack helpers
    def _stack_XY(pairs):
        if len(pairs) == 0:
            return (
                torch.empty(0, 0, 0, dtype = torch.float32),
                torch.empty(0, 0, 0, dtype = torch.long),
            )
        # Tuples of (C,T')
        Xs, Ys = zip(*pairs)
        # Ensure float/long dtypes
        Xs = [x.float() for x in Xs]
        Ys = [y.long()  for y in Ys]
        # (B,C,T')
        X = torch.stack(Xs, dim = 0)
        # (B,C,T')
        Y = torch.stack(Ys, dim = 0)
        return X, Y

    Xtrain, ytrain = _stack_XY(trainS)
    Xval, yval = _stack_XY(valS)
    Xtest, ytest = _stack_XY(testS)

    return {
        "train": {"X": Xtrain, "y": ytrain},
        "val":   {"X": Xval,   "y": yval},
        "test":  {"X": Xtest,  "y": ytest},
    }

def makeKFoldPatientTensors(df, fs, k: int = 10, valFrac: float = 0.10):
    """
    Deterministic patient-wise K-Fold (no shuffle), returning BOTH ID splits and tensors.

    Yields
    ------
    FoldIdx : int
    Payload : dict
      {
        "Splits": {
          "train": list[int],
          "val":   list[int],
          "test":  list[int],
        },
        "Tensors": {
          "train": (Xtr, ytr),   # Xtr: FloatTensor [N, 1, L], ytr: LongTensor [N, L]
          "val":   (Xva, yva),
          "test":  (Xte, yte),
        }
      }
    """
    patients = np.array(sorted(df["record_number"].unique()))
    if len(patients) < k:
        raise ValueError(f"Need at least {K} patients, got {len(patients)}.")

    folds = np.array_split(patients, k)

    for foldIdx in range(k):
        testIdx = foldIdx
        # Always the next 20 after test
        valIdx  = (foldIdx + 1) % k

        testPat = folds[testIdx].tolist()
        valPat  = folds[valIdx].tolist()
        trainPat = np.concatenate(
            [f for i, f in enumerate(folds) if i not in (testIdx, valIdx)]
        ).tolist()

        # Build tensors once here
        Xtr, ytr = gatherSplit(df, trainPat, fs, "train")
        Xva, yva = gatherSplit(df, valPat,   fs, "val")
        Xte, yte = gatherSplit(df, testPat,  fs, "test")

        yield foldIdx, {
            "Splits": {
                "train": list(map(int, trainPat)),
                "val":   list(map(int, valPat)),
                "test":  list(map(int, testPat)),
            },
            "Tensors": {
                "train": (Xtr, ytr),
                "val":   (Xva, yva),
                "test":  (Xte, yte),
            },
        }

def makeKFoldPatientTensors12Leads(df, fs: int = 500, trimSeconds: float = 1.0, k: int = 10, valFrac: float = 0.10):
    """
    Patient-wise K-Fold (no shuffling) for 12-lead ECG, returning BOTH ID splits and tensors.

    Steps:
      1) Build a cache of patient-level tensors via makeSamples().
      2) For each fold, stack cached tensors into (B, C, T') per split and yield.
    """
    
    # Build cache once
    sampleList = makeSamples(df, fs, trimSeconds)
    if not sampleList:
        raise ValueError("makeSamples produced no samples. Check df and trimSeconds/fs.")

    # Map patient → tensors
    cache = {int(recId): (X.float(), L.long()) for recId, X, L in sampleList}

    patients = np.array(sorted(cache.keys()))
    if len(patients) < k:
        raise ValueError(f"Need at least {k} patients, got {len(patients)}.")

    folds = np.array_split(patients, k)

    def _stack(idList):
        if len(idList) == 0:
            return (torch.empty((0, 0, 0), dtype = torch.float32),
                    torch.empty((0, 0, 0), dtype = torch.long))
        Xs, Ys = zip(*(cache[int(pid)] for pid in idList))
        return torch.stack(list(Xs), dim = 0), torch.stack(list(Ys), dim = 0)

    for foldIdx in range(k):
        testIdx = foldIdx
        # Always the next 20 after test
        valIdx  = (foldIdx + 1) % k

        testPat = folds[testIdx].tolist()
        valPat  = folds[valIdx].tolist()
        trainPat = np.concatenate(
            [f for i, f in enumerate(folds) if i not in (testIdx, valIdx)]
        ).tolist()
        
        Xtr, ytr = _stack(trainPat)
        Xva, yva = _stack(valPat)
        Xte, yte = _stack(testPat)

        yield foldIdx, {
            "Splits": {
                "train": list(map(int, trainPat)),
                "val":   list(map(int, valPat)),
                "test":  list(map(int, testPat)),
            },
            "Tensors": {
                "train": (Xtr, ytr),
                "val":   (Xva, yva),
                "test":  (Xte, yte),
            },
        }