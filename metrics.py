# metrics.py
import numpy as np
import torch
import torch.nn.functional as F

def segmentsFromLabels(labels, classId):
    """
    Return (starts, ends) sample indices for contiguous segments of a class.
    labels : 1D np.ndarray[int] of shape (L,)
    classId: int (e.g., 1=P, 2=QRS, 3=T)
    """
    y = (labels == classId).astype(np.int8)
    dy = np.diff(np.pad(y, (1, 1)))      # edges: +1 at start, -1 at end
    starts = np.where(dy == 1)[0]
    ends   = np.where(dy == -1)[0] - 1
    return starts, ends

def matchWithTolerance(gtPoints, prPoints, tolerance):
    """
    Greedy one-to-one matching between predicted and GT points within ±tolerance.
    Returns: TP, FP, FN, errors (signed: gt - pred, in samples)
    """
    gtPoints = np.asarray(gtPoints, dtype = int)
    prPoints = np.asarray(prPoints, dtype = int)
    usedGt = np.zeros(len(gtPoints), dtype = bool)
    errors = []
    TP = 0

    for p in prPoints:
        if gtPoints.size == 0:
            break
        diffs = gtPoints - p
        j = int(np.argmin(np.abs(diffs)))
        if not usedGt[j] and abs(diffs[j]) <= tolerance:
            usedGt[j] = True
            TP += 1
            errors.append(int(diffs[j]))

    FP = len(prPoints) - TP
    FN = len(gtPoints) - TP
    return TP, FP, FN, errors

@torch.no_grad()
def evalPaperMetrics(model, loader, fs = 500, tolerance = 150, device = None):
    """
    Works with:
      - fused outputs:    logits [B, K, T], targets [B, T]
      - per-lead outputs: logits [B, C, K, T], targets [B, C, T]
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    tol = int(round((tolerance / 1000.0) * fs))
    classIds = {"P": 1, "QRS": 2, "T": 3}
    results = {f"{w}_{b}": {"TP": 0, "FP": 0, "FN": 0, "errors": []}
               for w in classIds for b in ["on", "off"]}

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)

        # Convert targets to numpy
        if isinstance(yb, torch.Tensor):
            gts = yb.detach().cpu().numpy()
        else:
            gts = np.asarray(yb)

        if logits.dim() == 4:
            # per-lead: logits [B, C, K, T] -> preds [B, C, T]
            preds = logits.argmax(dim=2).detach().cpu().numpy()
            assert gts.ndim in (2, 3), "Expect targets [B,C,T] or [B,T]"
            if gts.ndim == 2:
                # If dataloader still provides [B,T], broadcast to all leads
                gts = np.repeat(gts[:, None, :], preds.shape[1], axis=1)

            B, C, T = preds.shape
            for b in range(B):
                for c in range(C):
                    pred = preds[b, c]
                    true = gts[b, c]
                    L = min(len(pred), len(true))
                    pred = pred[:L]; true = true[:L]

                    for waveName, cId in classIds.items():
                        gtOn, gtOff = segmentsFromLabels(true, cId)
                        prOn, prOff = segmentsFromLabels(pred, cId)

                        for btype, gtPts, prPts in (("on", gtOn, prOn), ("off", gtOff, prOff)):
                            TP, FP, FN, errs = matchWithTolerance(gtPts, prPts, tol)
                            key = f"{waveName}_{btype}"
                            results[key]["TP"] += TP
                            results[key]["FP"] += FP
                            results[key]["FN"] += FN
                            results[key]["errors"].extend(errs)

        elif logits.dim() == 3:
            # fused: logits [B, K, T] -> preds [B, T]
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            assert gts.ndim == 2, "Expect targets [B,T] for fused outputs"
            for pred, true in zip(preds, gts):
                L = min(len(pred), len(true))
                pred = pred[:L]; true = true[:L]

                for waveName, cId in classIds.items():
                    gtOn, gtOff = segmentsFromLabels(true, cId)
                    prOn, prOff = segmentsFromLabels(pred, cId)

                    for btype, gtPts, prPts in (("on", gtOn, prOn), ("off", gtOff, prOff)):
                        TP, FP, FN, errs = matchWithTolerance(gtPts, prPts, tol)
                        key = f"{waveName}_{btype}"
                        results[key]["TP"] += TP
                        results[key]["FP"] += FP
                        results[key]["FN"] += FN
                        results[key]["errors"].extend(errs)
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    # Aggregate to final metrics
    final = {}
    for key, vals in results.items():
        TP, FP, FN = vals["TP"], vals["FP"], vals["FN"]
        errs = np.array(vals["errors"], dtype=float) if vals["errors"] else np.array([])
        m = float(errs.mean()) if errs.size else 0.0
        sigma = float(errs.std(ddof=0)) if errs.size else 0.0
        Se = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        F1 = (2 * Se * PPV / (Se + PPV)) if (Se + PPV) > 0 else 0.0
        final[key] = {"m": m, "σ": sigma, "Se": Se, "PPV": PPV, "F1": F1}
    return final