import os
import math
import torch
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

def setGlobalSeed(seed: int = 42):
    """
    Sets the seed for reproducibility across:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and GPU)
    Also enforces deterministic behavior in CuDNN.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)  # Hash-based ops
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA determinism
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global seed set to {seed}")

def plotExampleWithMasks(df, recordNumber = None, channel = None):
    """
    Plots one ECG lead with shaded wave segments based on label_wave.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'record_number', 'channel', 'signal', and 'label_wave'.
    recordNumber : int, optional
        Patient record number to plot. If None, picks randomly.
    channel : str, optional
        Lead/channel name to plot. If None, picks randomly from the record.
    """

    if recordNumber is None:
        recordNumber = random.choice(df["record_number"].unique())

    subDf = df[df["record_number"] == recordNumber]

    if channel is None:
        channel = random.choice(subDf["channel"].unique())

    row = subDf[subDf["channel"] == channel].iloc[0]

    sig = row["signal"]
    mask = row["label_wave"]
    fs = row["fs"]
    t = np.arange(len(sig)) / fs

    plt.figure(figsize = (15, 4))
    plt.plot(t, sig, color = "black", linewidth = 1)

    # Shade P (1), QRS (2), T (3) segments
    colors = {1: "red", 2: "blue", 3: "green"}
    for label, color in colors.items():
        mask_indices = np.where(mask == label)[0]
        if mask_indices.size > 0:
            plt.fill_between(
                t, sig.min(), sig.max(),
                where = (mask == label),
                color = color, alpha = 0.3
            )

    plt.title(f"Record {recordNumber} | Channel {channel.upper()}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.show()

def plotLosses(train_losses, val_losses, title = "Loss", save_path=None):
    """
    Plot train/val loss from plain Python lists.
    - train_losses, val_losses: lists of floats
    - save_path: optional filepath to save the figure (e.g., 'logs/loss.png')
    """
    epochs_tr = list(range(1, len(train_losses) + 1))
    epochs_va = list(range(1, len(val_losses) + 1))

    plt.figure(figsize = (6, 4))
    if len(train_losses):
        plt.plot(epochs_tr, train_losses, label = "train")
    if len(val_losses):
        plt.plot(epochs_va, val_losses, label = "val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok = True)
        plt.savefig(save_path, dpi = 150)
    else:
        plt.show()

def saveBestModel(best_state_dict, path, epoch = None, val_loss = None):
    """
    Save the best model's state_dict (what you kept in bestState) to `path`.
    Stores optional metadata (epoch, val_loss).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"state_dict": best_state_dict}
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    if val_loss is not None:
        ckpt["val_loss"] = float(val_loss)
    torch.save(ckpt, path)

def loadBestModel(model, path, map_location = "cpu", strict = True):
    """
    Load weights from `path` into `model`. Returns (epoch, val_loss) if present.
    """
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt)  # supports raw state_dict or wrapped
    model.load_state_dict(state, strict=strict)
    return ckpt.get("epoch"), ckpt.get("val_loss")

def runEpoch(model, loader, criterion, optimizer = None, device = None, max_grad_norm = 5.0):
    """
    One pass over a DataLoader.
    - If optimizer is None -> eval mode (no grads). Otherwise -> train mode.
    - Expects model(x) -> [B, C, T] and labels y -> [B, T] (class indices).
    Returns: (avg_loss, token_accuracy)
    """
    if device is None:
        device = next(model.parameters()).device

    train = optimizer is not None
    model.train(train)

    totalLoss, totalTokens = 0.0, 0
    correct, tokens = 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking = True)
            yb = yb.to(device, non_blocking = True)

            if train:
                optimizer.zero_grad(set_to_none = True)

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            n_tok = yb.numel()
            totalLoss += float(loss) * n_tok
            totalTokens += n_tok

            preds = logits.argmax(dim = 1)
            correct += (preds == yb).sum().item()
            tokens += n_tok

    avgLoss = totalLoss / max(totalTokens, 1)
    avgAcc = correct / max(tokens, 1)
    return avgLoss, avgAcc

def runEpoch_MultiLead(model, loader, criterion, optimizer, device):
    isTrain = optimizer is not None
    dev = torch.device(device) if isinstance(device, str) else device
    model.train() if isTrain else model.eval()

    totalLoss, totalCorrect, totalCount = 0.0, 0, 0

    for x, y in loader:
        x = x.to(dev, dtype = torch.float32)   # x: [B,C,T]
        y = y.to(dev, dtype = torch.long)      # y: [B,C,T] (per-lead) or [B,T] (fused)

        if isTrain:
            optimizer.zero_grad()

        with torch.set_grad_enabled(isTrain):
            logits = model(x)
            if logits.dim() == 4:
                # per-lead path: logits [B,C,K,T], targets [B,C,T]
                B, C, K, T = logits.shape
                loss = criterion(logits.view(B * C, K, T), y.view(B * C, T))
                preds = logits.argmax(dim = 2)            # [B,C,T]
                totalCorrect += (preds == y).sum().item()
                totalCount  += y.numel()
            else:
                # fused path: logits [B,K,T], targets [B,T]
                B, K, T = logits.shape
                loss = criterion(logits, y)
                preds = logits.argmax(dim = 1)            # [B,T]
                totalCorrect += (preds == y).sum().item()
                totalCount  += y.numel()

            if isTrain:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        totalLoss += loss.item() * x.size(0)

    avgLoss = totalLoss / len(loader.dataset)
    acc = totalCorrect / totalCount if totalCount else 0.0
    return avgLoss, acc

def trainKFoldGeneric(
    FoldIter, ModelFn, RunEpochFn, *,
    Device = "cpu", Epochs = 50, BatchSize = 8, Lr = 1e-4, seed = 42,
    Criterion = None, optimName = 'Adam',
    NumWorkers = 0, PinMemory = False, PlotCols = 5,
    EvalFn = None, EvalKwargs = None,
    SaveFigPath = "kFold_Losses.png", SaveResultsPath = "kFold_Results"
):
    metrics_keys = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
    if EvalKwargs is None:
        EvalKwargs = {}

    foldsumm = []
    # per-DF rows
    f1_rows, ppv_rows, se_rows, err_rows = [], [], [], []

    foldlist = list(FoldIter)

    for FoldIdx, Payload in foldlist:
        (Xtr, ytr) = Payload["Tensors"]["train"]
        (Xva, yva) = Payload["Tensors"]["val"]
        (Xte, yte) = Payload["Tensors"]["test"]

        TrainLoader = DataLoader(TensorDataset(Xtr, ytr), batch_size = BatchSize, num_workers = NumWorkers, pin_memory = PinMemory)
        ValLoader = DataLoader(TensorDataset(Xva, yva), batch_size = BatchSize, num_workers = NumWorkers, pin_memory = PinMemory)
        TestLoader = DataLoader(TensorDataset(Xte, yte), batch_size = BatchSize, num_workers = NumWorkers, pin_memory = PinMemory)

        # Re-seed right before model init
        torch.manual_seed(seed)
        model = ModelFn().to(Device)
        if optimName == 'AdamW':
            opt = torch.optim.AdamW(model.parameters(), lr = Lr)
        else:
            opt = torch.optim.Adam(model.parameters(), lr = Lr)

        train_losses, val_losses = [], []
        best_val, best_ep = 1e9, 0
        best_state = None

        for ep in range(1, Epochs+1):
            tr_loss, _ = RunEpochFn(model, TrainLoader, Criterion, opt, Device)
            va_loss, _ = RunEpochFn(model, ValLoader, Criterion, None, Device)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            if va_loss < best_val:
                best_val, best_ep = va_loss, ep
                best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}

            if ep % max(1, Epochs // 5) == 0 or ep == 1:
                print(f"[Fold {FoldIdx + 1}/{len(foldlist)}] Ep {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        model.load_state_dict(best_state, strict = True)
        metrics = EvalFn(model, TestLoader, **EvalKwargs)

        # Build per-fold rows for each DF
        row_f1  = {"Fold": FoldIdx+1}
        row_ppv = {"Fold": FoldIdx+1}
        row_se  = {"Fold": FoldIdx+1}
        row_err = {"Fold": FoldIdx+1}

        for k in metrics_keys:
            row_f1[k]  = float(metrics[k]["F1"])
            row_ppv[k] = float(metrics[k]["PPV"])
            row_se[k]  = float(metrics[k]["Se"])
            # store mean (m) and std (σ) as separate numeric columns
            row_err[f"{k}_m"]  = float(metrics[k]["m"])
            row_err[f"{k}_sd"] = float(metrics[k]["σ"])

        f1_rows.append(row_f1)
        ppv_rows.append(row_ppv)
        se_rows.append(row_se)
        err_rows.append(row_err)

        foldsumm.append({
            "FoldIdx": FoldIdx,
            "TrainLosses": train_losses,
            "ValLosses": val_losses,
            "BestEp": best_ep,
            "BestState": best_state
        })

        del model
        torch.cuda.empty_cache()

    # Plot losses
    K = len(foldlist)
    cols = PlotCols
    rows = math.ceil(K/cols)
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 3), squeeze = False)
    for i, info in enumerate(foldsumm):
        r,c = divmod(i, cols)
        ax = axes[r][c]
        ax.plot(info["TrainLosses"], label = "train")
        ax.plot(info["ValLosses"], label = "val")
        ax.set_title(f"Fold {info['FoldIdx'] + 1} (best@{info['BestEp']})")
        ax.legend()
    for j in range(K, rows*cols):
        r,c = divmod(j, cols); axes[r][c].axis("off")
    plt.tight_layout()
    plt.savefig(SaveFigPath, dpi = 150)
    plt.show()

    # Build DataFrames + Mean row
    df_f1  = pd.DataFrame(f1_rows)
    df_ppv = pd.DataFrame(ppv_rows)
    df_se  = pd.DataFrame(se_rows)
    df_err = pd.DataFrame(err_rows)

    # Append per-column means as last row
    mean_row_f1  = {"Fold": "Mean"} | {k: df_f1[k].mean() for k in metrics_keys}
    mean_row_ppv = {"Fold": "Mean"} | {k: df_ppv[k].mean() for k in metrics_keys}
    mean_row_se  = {"Fold": "Mean"} | {k: df_se[k].mean() for k in metrics_keys}

    err_cols = [f"{k}_m" for k in metrics_keys] + [f"{k}_sd" for k in metrics_keys]
    mean_row_err = {"Fold": "Mean"} | {c: df_err[c].mean() for c in err_cols}

    df_f1 = pd.concat([df_f1,  pd.DataFrame([mean_row_f1])],  ignore_index = True)
    df_ppv = pd.concat([df_ppv, pd.DataFrame([mean_row_ppv])], ignore_index = True)
    df_se = pd.concat([df_se,  pd.DataFrame([mean_row_se])],  ignore_index = True)
    df_err = pd.concat([df_err, pd.DataFrame([mean_row_err])], ignore_index = True)

    # Print quick views
    print("\nF1 by fold + mean:")
    print(df_f1.round(4).to_string(index = False))
    print("\nPrecision (PPV) by fold + mean:")
    print(df_ppv.round(4).to_string(index = False))
    print("\nRecall (Se) by fold + mean:")
    print(df_se.round(4).to_string(index = False))
    print("\nMean error (m) and Std (sd) by fold + mean:")
    print(df_err.round(2).to_string(index = False))

    # Save CSVs
    df_f1.to_csv (f"{SaveResultsPath}_F1.csv", index = False)
    df_ppv.to_csv(f"{SaveResultsPath}_Precision.csv", index = False)
    df_se.to_csv (f"{SaveResultsPath}_Recall.csv", index = False)
    df_err.to_csv(f"{SaveResultsPath}_Error.csv", index = False)

    # Save checkpoints summary for reproducibility
    with open(f"{SaveResultsPath}_CheckPoints.pkl", "wb") as f:
        pickle.dump(foldsumm, f)

    # Return folds summary + 4 DFs
    return foldsumm, df_f1, df_ppv, df_se, df_err

def plotPredsVsGt(preds, df, recordNumber, lead, config, modelName = "UNet1D"):
    """
    Args:
        preds: 1D torch.Tensor or np.ndarray of class ids [L_pred]
        df: pandas DataFrame with columns ['record_number','channel','signal','label_wave']
        recordNumber: int (e.g., 181)
        lead: str (e.g., 'ii')
        config: dict with config['paths']['results_dir']
        modelName: str, used in output filename

    Returns:
        acc (float), savePath (str)
    """

    # --- grab signal + ground truth from recordDf ---
    sel = df[(df["record_number"] == recordNumber) & (df["channel"] == lead)]
    if len(sel) == 0:
        raise ValueError(f"No rows found for record_number = {recordNumber}, channel = '{lead}'")

    sig = sel["signal"].iat[0]
    gt = sel["label_wave"].iat[0]

    # --- to tensors (keep device flexible; preds may already be on GPU) ---
    if torch.is_tensor(preds):
        predsT = preds
    else:
        predsT = torch.as_tensor(preds, dtype = torch.long)

    gtT = torch.as_tensor(gt, dtype = predsT.dtype, device = predsT.device)
    if gtT.dim() == 2 and gtT.size(0) == 1:  # handle [1, L]
        gtT = gtT.squeeze(0)

    # --- accuracy on overlap ---
    L_pred = predsT.numel()
    L_gt = gtT.numel()
    L_overlap = min(L_pred, L_gt)
    acc = (predsT[:L_overlap] == gtT[:L_overlap]).float().mean().item()
    print(f"token-acc: {acc:.3f}")

    # --- to numpy for plotting ---
    # signal might be torch or np or list; convert safely
    sigT = torch.as_tensor(sig, dtype = torch.float32, device = predsT.device)
    sigNp = sigT.detach().cpu().numpy()
    predsNp = predsT.detach().cpu().numpy()
    gtNp = gtT.detach().cpu().numpy()

    # use common x-axis length conservatively to avoid shape mismatches
    L_sig = len(sigNp)
    L_plot = min(L_sig, len(predsNp), len(gtNp))
    x = np.arange(L_plot)

    # --- make figure ---
    fig, (axGt, axPred) = plt.subplots(2, 1, figsize = (12, 6), sharex = True, constrained_layout = True)

    # TOP: ECG + Ground Truth
    axGt.plot(x, sigNp[:L_plot], color = "black", linewidth = 1, label = "ECG")
    axGt.set_ylabel("amplitude")
    axGt.set_title(f"Ground Truth | rec {recordNumber} | lead {lead}")

    axGtR = axGt.twinx()
    axGtR.step(x, gtNp[:L_plot], where = "post", color = "green", alpha = 0.9, label = "GT")
    axGtR.set_ylim(-0.5, 3.5)
    axGtR.set_yticks([0, 1, 2, 3])
    axGtR.set_ylabel("class")
    axGt.legend(loc = "upper left"); axGtR.legend(loc = "upper right")

    # BOTTOM: ECG + Prediction
    axPred.plot(x, sigNp[:L_plot], color = "black", linewidth = 1, label = "ECG")
    axPred.set_ylabel("amplitude")
    axPred.set_xlabel("sample")
    axPred.set_title("Prediction")

    axPredR = axPred.twinx()
    axPredR.step(x, predsNp[:L_plot], where = "post", color = "red", alpha = 0.9, label = "Pred")
    axPredR.set_ylim(-0.5, 3.5)
    axPredR.set_yticks([0, 1, 2, 3])
    axPredR.set_ylabel("class")
    axPred.legend(loc = "upper left"); axPredR.legend(loc = "upper right")

    savePath = config["paths"]["results_dir"] + f"{modelName}_Patient_{recordNumber}_Lead_{str(lead).upper()}_split.png"
    fig.savefig(savePath, dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close(fig)
    return acc, savePath