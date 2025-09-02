import os
import glob
import yaml
import wfdb
import numpy as np
import pandas as pd

def loadRecords(location, recordNumber):
    data = []
    # Load full multi-lead ECG record
    record = wfdb.rdrecord(f"{location}/{recordNumber}")
    fs = record.fs
    signalLength = record.p_signal.shape[0]
    # Loop through all channels (leads)
    for i, channel in enumerate(record.sig_name):
        try:
            # Load annotations for that specific lead
            ann = wfdb.rdann(f"{location}/{recordNumber}", channel)
            annSymbols = ann.symbol
            annSamples = ann.sample
            numAnnotation = len(annSymbols)
        except:
            print(f"Annotation for channel '{channel}' not found.")
            annSamples = []
            numAnnotation = 0
        # Extract signal for the current channel
        signal = record.p_signal[:, i]
        # Append structured info
        data.append([
            recordNumber,     # Record Number
            channel,          # Channel Name
            fs,               # Frequency / Sampling Rate
            i,                # Lead Index
            signal,           # Signal
            len(signal),      # Signal Length
            annSymbols,       # Annotation
            annSamples,       # Annotation Samples
            numAnnotation     # Number of Annotations
        ])
    return data

# ---- Annotation validation ----
def isValidAnnotationFormat(annSymbols):
    """
    Checks if annotation symbols follow the expected format: (wave)
    wave ∈ {'p', 'N', 't'}
    """
    waves = {'p', 'N', 't'}
    if len(annSymbols) % 3 != 0:
        return False
    
    for i in range(0, len(annSymbols), 3):
        if annSymbols[i] != '(' or annSymbols[i+1] not in waves or annSymbols[i+2] != ')':
            return False
    return True

def fixAnnotationIssues(annSymbols, annSamples):
    """
    Fixes common formatting issues in ECG annotation symbols and their corresponding sample indices.

    The expected format for annotations is triplets: (wave), 
    where 'wave' ∈ {'p', 'N', 't'}. Each triplet represents 
    the start, type, and end marker for a wave in the ECG signal.

    This function:
    - Converts NumPy arrays to Python lists for easier modification.
    - Identifies and corrects misordered triplets (e.g., wave, '(', ')').
    - Inserts missing '(' or ')' symbols where necessary.
    - Adjusts corresponding sample indices to match inserted symbols.
    - Handles empty parentheses blocks and other structural anomalies.
    - Converts lists back to NumPy arrays before returning.

    Parameters
    ----------
    annSymbols : list or np.ndarray
        Sequence of annotation symbols for an ECG lead.
    annSamples : list or np.ndarray
        Corresponding sample indices for each annotation symbol.

    Returns
    -------
    annSymbols : np.ndarray
        Corrected sequence of annotation symbols.
    annSamples : np.ndarray
        Corrected sequence of annotation sample indices.
    """
    
    waves = {'p', 'N', 't'}
    
    # Convert to lists if they are np.ndarray
    annSymbols = annSymbols.tolist() if isinstance(annSymbols, np.ndarray) else annSymbols
    annSamples = annSamples.tolist() if isinstance(annSamples, np.ndarray) else annSamples

    i = 0
    while (i < len(annSymbols)):
        # Case 1: Proper pattern
        if ((i + 2 < len(annSymbols)) and (annSymbols[i] == '(' and annSymbols[i+1] in waves and annSymbols[i+2] == ')')):
            i += 3
        # Case 2: wave, (, ) -> (, wave, )
        elif ((i + 2 < len(annSymbols)) and (annSymbols[i] in waves and annSymbols[i+1] == '(' and annSymbols[i+2] == ')')):
            annSymbols[i], annSymbols[i+1] = annSymbols[i+1], annSymbols[i]
            i += 3
        # Case 3: (, ), wave -> (, wave, )
        elif ((i + 2 < len(annSymbols)) and (annSymbols[i] == '(' and annSymbols[i+1] == ')' and annSymbols[i+2] in waves)):
            annSymbols[i+1], annSymbols[i+2] = annSymbols[i+2], annSymbols[i+1]
            i += 3
        # Case 4: (, ), ( — probably empty () block
        elif ((i + 2 < len(annSymbols)) and (annSymbols[i] == '(' and annSymbols[i+1] == ')' and annSymbols[i+2] == '(')):
            # Could infer based on context — left for next step
            i += 2
        # Case 5: wave, ) — missing '('
        elif annSymbols[i] in waves and annSymbols[i+1] == ')':
            if((i + 2 < len(annSymbols) and annSymbols[i+2] != ')') or (i+2 == len(annSymbols))):
                annSymbols.insert(i, '(')
                annSamples.insert(i, annSamples[i] - 1)
                i += 3
            else:
                i += 1
        # Case 6: (, wave — missing ')'
        elif annSymbols[i] == '(' and annSymbols[i+1] in waves and annSymbols[i+2] == '(':
            annSymbols.insert(i+2, ')')
            annSamples.insert(i+2, annSamples[i+1] + 1)
            i += 3
        else:
            i += 1

    # Convert back to np.ndarray if needed
    annSymbols = np.array(annSymbols)
    annSamples = np.array(annSamples)
    return annSymbols, annSamples

def isAllAnnotationValids(annChannelDictionary):
    """
    Validates annotation formats for all channels in an ECG record.

    Each channel's annotation sequence is checked against the expected 
    triplet pattern: (wave), where 'wave' ∈ {'p', 'N', 't'}.

    Parameters
    ----------
    annChannelDictionary : dict
        Dictionary mapping channel names to their annotation symbol sequences.

    Returns
    -------
    str or None
        If invalid annotations are found, returns a message listing the 
        affected channels. Returns None if all annotations are valid.
    """
    annLengths = []
    invalidChannels = []
    for channel in annChannelDictionary.keys():
        annLengths.append(len(annChannelDictionary[channel]))
        isValid = isValidAnnotationFormat(annChannelDictionary[channel])
        if not isValid:
            invalidChannels.append(channel)

    if invalidChannels:
        return f"Triplet issue in annotation(s) in channel(s): {'-'.join(invalidChannels)}"
    return None

def isValidSamples(samples):
    """
    Checks if annotation sample indices are strictly increasing.

    Parameters
    ----------
    samples : list or np.ndarray
        Sequence of annotation sample indices.

    Returns
    -------
    bool
        True if all sample indices are strictly increasing, 
        False if any index is equal to or less than the previous one.
    """
    current = samples[0]
    for i in range(1, len(samples)):
        if samples[i] <= current:
            return False
        current = samples[i]
    return True

def createLabelsFromAnnotations(annSymbols, annSamples, signalLength):
    """
    Creates per-sample label arrays from ECG wave annotations.

    Generates two label sequences:
    - label_fidu: Labels only the fiducial point (peak) of each wave.
    - label_wave: Labels the entire segment from wave start to end.

    Parameters
    ----------
    annSymbols : list or np.ndarray
        Sequence of annotation symbols in the format (wave), where
        'wave' ∈ {'p', 'N', 't'}.
    annSamples : list or np.ndarray
        Corresponding sample indices for each annotation symbol.
    signalLength : int
        Total length of the ECG signal (number of samples).

    Returns
    -------
    label_fidu : np.ndarray
        1D array of length `signalLength`, dtype uint8.
        Fiducial labels: 0 for non-peak samples, 
        1 for P peak, 2 for QRS peak, 3 for T peak.
    label_wave : np.ndarray
        1D array of length `signalLength`, dtype uint8.
        Wave segment labels: 0 for background,
        1 for P segment, 2 for QRS segment, 3 for T segment.

    Notes
    -----
    - Indexing is clipped to the signal length to avoid out-of-bounds errors.
    - This function assumes triplets in the form:
      '(' → wave symbol → ')'.
    """

    label_fidu = np.zeros(signalLength, dtype = np.uint8)
    label_wave = np.zeros(signalLength, dtype = np.uint8)

    waveMap = {'p': 1, 'N': 2, 't': 3}
    i = 0
    while(i < len(annSymbols) - 2):
        # Looking for triplet: '(', symbol, ')'
        if (annSymbols[i] == '(' and annSymbols[i+2] == ')'):
            waveType = annSymbols[i+1]
            if(waveType in waveMap):
                waveLabel = waveMap[waveType]
                start = annSamples[i]
                peak = annSamples[i+1]
                end = annSamples[i+2]

                # Safe indexing
                start = max(0, min(signalLength - 1, start))
                peak = max(0, min(signalLength - 1, peak))
                end = max(0, min(signalLength - 1, end))

                # Assign to wave label (segment)
                label_wave[start : end + 1] = waveLabel
                
                # Assign to fiducial label (just peak)
                label_fidu[peak] = waveLabel
                
            # Move to next triplet
            i += 3
        else:
            # Move forward if not a triplet
            i += 1
    return label_fidu, label_wave

def findIdx(dfLocal, recordNumber, lead):
    """
    Find the index of a specific record and lead in the given DataFrame.

    Parameters
    ----------
    dfLocal : pandas.DataFrame
        DataFrame containing ECG annotations and samples. 
        Must include columns 'record_number' and 'channel'.
    recordNumber : int
        The record number to search for.
    lead : str
        The channel/lead name (e.g., 'V1', 'V2', 'V5'). 
        Matching is case-insensitive.

    Returns
    -------
    int or None
        The index of the matching row in dfLocal if found, 
        otherwise None.
    """
    
    mask = (dfLocal["record_number"] == recordNumber) & (dfLocal["channel"].str.lower() == lead.lower())
    idx = dfLocal[mask].index
    return idx[0] if len(idx) else None

def applyManualAnnotationFixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply targeted manual corrections to specific annotation errors 
    in the ECG dataset.

    The function performs the following three fixes:

    1. Record 8, channel 'V5':
       - Insert an 'N' annotation at position 7 in 'annotation'.
       - Insert the midpoint sample between positions 6 and 7 in 'samples'.

    2. Record 116, channel 'V1':
       - Replace the annotation slice [6:24] with a valid '( t )' triplet.
       - Onset set to sample[6], offset set to sample[23], 
         and peak set to the integer midpoint between them.

    3. Record 116, channel 'V2':
       - Replace the annotation slice [6:18] with a valid '( t )' triplet.
       - Onset set to sample[6], offset set to sample[17], 
         and peak set to the integer midpoint between them.

    Notes
    -----
    - Channel matching is case-insensitive.
    - If the target row is missing or has too few elements, the fix is skipped.
    - 'samples' remain as int64 arrays.
    - 'annotation' is stored as a NumPy array of strings.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing ECG records. 
        Must include the columns: 
        ['record_number', 'channel', 'annotation', 'samples'].

    Returns
    -------
    pandas.DataFrame
        The same DataFrame (modified in place) returned for convenience.
    """

    # --- Fix 1: (8, 'v5') insert 'N' and midpoint sample at position 7 ---
    idx = findIdx(df, 8, "v5")
    if idx is not None:
        annList = list(df.at[idx, "annotation"])
        sampleList = list(df.at[idx, "samples"])
        if len(annList) >= 7 and len(sampleList) >= 8:
            annList.insert(7, "N")
            mid = int((sampleList[6] + sampleList[7]) / 2)
            sampleList.insert(7, mid)
            df.at[idx, "annotation"] = np.array(annList, dtype = object)
            df.at[idx, "samples"] = np.array(sampleList, dtype = "int64")

    # --- Fix 2: (116, 'v1') replace 6:24 with '( t )' triplet ---
    idx = findIdx(df, 116, "v1")
    if idx is not None:
        annList = list(df.at[idx, "annotation"])
        annList = annList[0:6] + [np.str_('('), np.str_('t'), np.str_(')')] + annList[24:]
        sampleList = list(df.at[idx, "samples"])
        sampleList = sampleList[0:6] + [sampleList[6], int((sampleList[6] + sampleList[23])/2), sampleList[23]] + sampleList[24:]
        df.at[idx, "annotation"] = np.array(annList, dtype = object)
        df.at[idx, "samples"] = np.array(sampleList, dtype = "int64")

    # --- Fix 3: (116, 'v2') replace 6:18 with '( t )' triplet ---
    idx = findIdx(df, 116, "v2")
    if idx is not None:
        annList = list(df.at[idx, "annotation"])
        annList = annList[0:6] + [np.str_('('), np.str_('t'), np.str_(')')] + annList[18:]
        sampleList = list(df.at[idx, "samples"])
        sampleList = sampleList[0:6] + [sampleList[6], int((sampleList[6] + sampleList[17])/2), sampleList[17]] + sampleList[18:]
        df.at[idx, "annotation"] = np.array(annList, dtype = object)
        df.at[idx, "samples"] = np.array(sampleList, dtype = "int64")
        
    return df

def dataPipeline(configPath: str = "configs.yaml", recordNumbers = None) -> pd.DataFrame:
    """
    End-to-end ECG data preprocessing pipeline.

    Order of operations:
      1) Load records from raw_data_dir (configs.yaml)
      2) AUTO-FIX: validate & fix annotations/samples programmatically
      3) MANUAL FIX: apply targeted edits via applyManualAnnotationFixes(df)
      4) LABELS: generate label_fidu and label_wave from (possibly edited) annotations
      5) Sort by record_number, lead_index and return
    """
    
    # Read config for raw data path
    with open(configPath, "r") as f:
        cfg = yaml.safe_load(f)
    rawDir = cfg["paths"]["raw_data_dir"]

    # Discover records if not provided
    if recordNumbers is None:
        hea_files = glob.glob(os.path.join(rawDir, "*.hea"))
        recordNumbers = [os.path.splitext(os.path.basename(p))[0] for p in hea_files]

    rows = []
    for rec in recordNumbers:
        # loadRecords returns a list of per-lead rows:
        # [recordNumber, channel, fs, leadIndex, signal, signalLength, annSymbols, annSamples, numAnnotation]
        leadRows = loadRecords(rawDir.rstrip("/"), str(rec))
        for item in leadRows:
            recordNumber, channel, fs, leadIndex, signal, signalLength, annSymbols, annSamples, numAnnotation = item

            # Ensure numpy arrays
            annSymbols = np.asarray(annSymbols)
            annSamples = np.asarray(annSamples, dtype = int)

            rows.append({
                "record_number": int(recordNumber),
                "channel": channel,
                "fs": fs,
                "lead_index": leadIndex,
                "signal": signal,
                "signal_length": int(signalLength),
                "annotation": annSymbols,
                "samples": annSamples,
                "num_annotations": int(numAnnotation),
                "label_fidu": None,
                "label_wave": None,
            })

    df = pd.DataFrame(rows)

    # AUTO-FIX: validate & fix programmatically
    for idx, row in df.iterrows():
        annSymbols = row["annotation"]
        annSamples = row["samples"]

        needsFix = False
        if isinstance(annSymbols, np.ndarray) and annSymbols.size and not isValidAnnotationFormat(annSymbols):
            needsFix = True
        if isinstance(annSamples, np.ndarray) and annSamples.size and not isValidSamples(annSamples):
            needsFix = True

        if needsFix and annSymbols.size and annSamples.size:
            annSymbols, annSamples = fixAnnotationIssues(annSymbols, annSamples)
            df.at[idx, "annotation"] = annSymbols
            df.at[idx, "samples"] = annSamples
            df.at[idx, "num_annotations"] = len(annSymbols)

    # Keep the rawDf
    rawDf = df.copy(deep = True)
    
    # MANUAL FIX after auto-fix
    df = applyManualAnnotationFixes(df)

    # LABELS from (possibly edited) annotations
    for idx, row in rawDf.iterrows():
        annSymbols = row["annotation"]
        annSamples = row["samples"]
        sigLen = int(row["signal_length"])

        if isinstance(annSymbols, np.ndarray) and isinstance(annSamples, np.ndarray) \
           and annSymbols.size >= 3 and annSamples.size >= 3:
            labelFidu, labelWave = createLabelsFromAnnotations(annSymbols, annSamples, sigLen)
        else:
            labelFidu = np.zeros(sigLen, dtype = np.uint8)
            labelWave = np.zeros(sigLen, dtype = np.uint8)

        rawDf.at[idx, "label_fidu"] = labelFidu
        rawDf.at[idx, "label_wave"] = labelWave

    # LABELS from (possibly edited) annotations
    for idx, row in df.iterrows():
        annSymbols = row["annotation"]
        annSamples = row["samples"]
        sigLen = int(row["signal_length"])

        if isinstance(annSymbols, np.ndarray) and isinstance(annSamples, np.ndarray) \
           and annSymbols.size >= 3 and annSamples.size >= 3:
            labelFidu, labelWave = createLabelsFromAnnotations(annSymbols, annSamples, sigLen)
        else:
            labelFidu = np.zeros(sigLen, dtype = np.uint8)
            labelWave = np.zeros(sigLen, dtype = np.uint8)

        df.at[idx, "label_fidu"] = labelFidu
        df.at[idx, "label_wave"] = labelWave

    # Sort & Return 
    df = df.sort_values(by = ["record_number", "lead_index"], ascending = [True, True]).reset_index(drop = True)
    return rawDf, df