import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConstantPad1d((4, 4), 0), # padding = 4 for k = 9
            nn.Conv1d(in_channels, out_channels, kernel_size = 9),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace = True)
        )
    def forward(self, x): return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, up_channels, skip_channels, out_channels):
        """
        in_channels: channels of the incoming decoder feature
        up_channels: channels after transposed conv (before concat)
        skip_channels: channels from the encoder skip
        out_channels: channels after the two convs in this block
        """
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, up_channels, kernel_size = 8, stride = 2, padding = 3)
        self.conv1 = ConvBNReLU(up_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, skip, x):
        x = self.upconv(x)
        # align lengths (odd/even effects cause 1-sample mismatches)
        diff = skip.size(-1) - x.size(-1)
        if diff > 0: # x shorter → pad
            x = F.pad(x, (0, diff))
        elif diff < 0: # x longer → crop
            x = x[:, :, :skip.size(-1)]
        x = torch.cat([skip, x], dim = 1)  # concat → widths: 96 / 48 / 24 / 12
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet1D(nn.Module):
    def __init__(self, input_channels = 1, num_classes = 4):
        super().__init__()
        # stem: 1 → 4 → 4
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.init_conv1 = ConvBNReLU(input_channels, 4)
        self.init_conv2 = ConvBNReLU(4, 4)

        # Encoder: 4 → 8 → 16 → 32 → 64
        self.down1 = DownBlock(4, 8)
        self.down2 = DownBlock(8, 16)
        self.down3 = DownBlock(16, 32)
        self.down4 = DownBlock(32, 64)

        # Decoder, Upconv outputs: 64, 32, 16, 8 (so 64 + 32 = 96, 32 + 16 = 48, 16 + 8 = 24, 8 + 4 = 12)
        self.up1 = UpBlock(in_channels = 64, up_channels = 64, skip_channels = 32, out_channels = 32)  # 96 → 32
        self.up2 = UpBlock(in_channels = 32, up_channels = 32, skip_channels = 16, out_channels = 16)  # 48 → 16
        self.up3 = UpBlock(in_channels = 16, up_channels = 16, skip_channels = 8,  out_channels = 8)   # 24 → 8
        self.up4 = UpBlock(in_channels = 8,  up_channels = 8,  skip_channels = 4,  out_channels = 4)   # 12 → 4

        # Output: 4 logits
        self.final_conv = nn.Conv1d(4, num_classes * input_channels, kernel_size = 1)

    def forward(self, x):
        # stem
        x0 = self.init_conv2(self.init_conv1(x))   # 4
        # encoder
        x1 = self.down1(x0)  # 8
        x2 = self.down2(x1)  # 16
        x3 = self.down3(x2)  # 32
        x4 = self.down4(x3)  # 64
        # decoder
        x = self.up1(x3, x4) # concat 96 → 32
        x = self.up2(x2, x)  # concat 48 → 16
        x = self.up3(x1, x)  # concat 24 → 8
        x = self.up4(x0, x)  # concat 12 → 4

        logits = self.final_conv(x)

        if(self.input_channels > 1):
            B, CK, L = logits.shape
            C = self.input_channels
            K = self.num_classes
            assert CK == C * K, f"Expected {C * K} channels, got {CK}"
            # (B, C, K, L)
            return logits.view(B, C, K, L).contiguous()
        # (B, K, L)
        return logits

class CNNBiLSTM_Model_Profile11(nn.Module):
    def __init__(self, input_channels = 1, num_classes = 4, lstm_hidden = 128, lstm_layers = 1):
        super().__init__()

        # 7 explicit Conv + ReLU layers
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv1 = self._seqConvBlock(input_channels, 8)
        self.conv2 = self._seqConvBlock(8, 16)
        self.conv3 = self._seqConvBlock(16, 32)
        self.conv4 = self._seqConvBlock(32, 64)
        self.conv5 = self._seqConvBlock(64, 128)
        self.conv6 = self._seqConvBlock(128, 256)
        self.conv7 = self._seqConvBlock(256, 512)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size = 512,
            hidden_size = lstm_hidden,
            num_layers = lstm_layers,
            batch_first = True,
            bidirectional = True
        )

        # Classifier
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes * input_channels)

    def _seqConvBlock(self, inputChannels, outChannels, kernel = 3, stride = 1, padding = 1):
        return nn.Sequential(
            nn.Conv1d(inputChannels, outChannels, kernel_size = kernel, stride = stride, padding = 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, C, L)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # BiLSTM expects (B, L, C)
        # (B, L, 512)
        x = x.transpose(1, 2)
        # (B, L, 2 * hidden)
        x, _ = self.bilstm(x)

        # (B, L, K) or (B, L, C * K)
        logits = self.classifier(x)
        if self.input_channels > 1:
            B, L, CK = logits.shape
            C = self.input_channels
            K = self.num_classes
            assert CK == C * K, f"Expected {C * K}, got {CK}"
            # (B, C, K, L)
            return logits.view(B, L, C, K).permute(0, 2, 3, 1).contiguous()
        # (B, K, L)
        return logits.transpose(1, 2).contiguous()  

class PatchEmbed1D(nn.Module):
    """
    Depthwise 1D patch embedding for multi-lead ECG (or any C × T signal).

    Converts an input tensor of shape **[B, C, T]** into per-lead, per-patch
    embeddings **[B, C, N, D]** using a grouped 1D convolution:
    - Each lead is processed independently (groups = C).
    - Kernel size = `patchSize` (patch length in samples).
    - Stride = `patchSize` (non-overlapping patches).
    - For each lead, `embedDim` filters produce a D-dim embedding per patch.

    Parameters
    ----------
    numLeads : int
        Number of input channels/leads (C).
    embedDim : int, default = 160
        Embedding dimension per lead per patch (D).
    patchSize : int, default = 40
        Patch length in samples; also used as the stride (no overlap).
    stride : int or None, default = None
        (Ignored in current implementation; stride is set to `patchSize`.)

    Input shape
    -----------
    x : torch.Tensor
        [B, C, T] — batch size B, leads C, length T (T ≥ patchSize).

    Output shape
    ------------
    torch.Tensor
        [B, C, N, D] where
        N = floor((T - patchSize) / patchSize) + 1  (no padding).

    Notes
    -----
    - This is a **depthwise** conv over time (groups = C). The internal conv
      has weight shape [(C * D), 1, patchSize]; each lead gets its own D filters.
    - No padding is applied; if T is not an exact multiple of `patchSize`,
      the tail < `patchSize` is dropped by the convolution.
    - Positional or lead embeddings are **not** added here; this layer only
      produces patch embeddings.
    """
    
    # [B, C, T] -> [B, C, N, D]
    def __init__(self, numLeads, embedDim = 160, patchSize = 40, stride = None):
        super().__init__()
        self.numLeads = numLeads
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.stride = patchSize
        self.proj = nn.Conv1d(
            numLeads, numLeads * embedDim,
            kernel_size = patchSize,
            stride = self.stride, 
            groups = numLeads
        )

    def forward(self, x):
        # x: [B, C, T]
        # [B, C * D, N]
        z = self.proj(x)
        B, _, N = z.shape
        # [B, C, N, D]
        z = z.view(B, self.numLeads, self.embedDim, N).transpose(2, 3).contiguous()
        return z

class CCTBlock(nn.Module):
    """
    Criss-Cross Transformer block for ECG.
    Input:  z [B, C, N, D] 
        B = batch
        C = number of leads (channels)
        N = number of tokens (time patches)
        D = embedding dimension
    Operations:
      - Temporal attention (per lead, across tokens)
      - Spatial attention (per token, across leads)
      - Fuse outputs (concat → linear proj)
      - MLP with residual
    Output: same shape [B, C, N, D]
    """
    def __init__(self, dim: int, numHeads: int = 4, mlpRatio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        assert numHeads % 2 == 0, "Use an even number of heads (split temp/spat)."

        # Norms for attention
        self.normTemporal = nn.LayerNorm(dim)
        self.normSpatial = nn.LayerNorm(dim)

        # Temporal attention: per lead → heads/2
        self.tempAttention = nn.MultiheadAttention(
            embed_dim = dim, num_heads = numHeads // 2, dropout = dropout, batch_first = True
        )

        # Spatial attention: per token → heads/2
        self.spatAttention = nn.MultiheadAttention(
            embed_dim = dim, num_heads = numHeads // 2, dropout = dropout, batch_first = True
        )

        # Fuse (temporal + spatial) features
        self.fuse = nn.Linear(2 * dim, dim)

        # Norm + MLP
        self.normMLP = nn.LayerNorm(dim)
        hidden = int(dim * mlpRatio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C, N, D]
        """
        B, C, N, D = z.shape

        # Temporal attention (across tokens per lead)
        # [B * C, N, D]
        xt = self.normTemporal(z).reshape(B * C, N, D)
        # [B * C, N, D]
        t_out, _ = self.tempAttention(xt, xt, xt)
        t_out = t_out.reshape(B, C, N, D)

        # Spatial attention (across leads per token)
        # [B * N, C, D]
        xs = self.normSpatial(z).permute(0, 2, 1, 3).reshape(B * N, C, D)
        # [B * N, C, D]
        s_out, _ = self.spatAttention(xs, xs, xs)
        s_out = s_out.reshape(B, N, C, D).permute(0, 2, 1, 3).contiguous()

        # Fuse + Residual
        # [B, C, N, 2D]
        fused = torch.cat([t_out, s_out], dim = -1)
        # [B, C, N, D]
        fused = self.fuse(fused)
        z = z + fused
        
        # MLP + Residual
        z = z + self.mlp(self.normMLP(z))
        return z


class CC_Transformer(nn.Module):
    """
    Criss-Cross Transformer for ECG segmentation.

    Input: x [B, C, T]
    Output: logits [B, C, K, T]
    """
    def __init__(
        self, numLeads: int = 12, numClasses: int = 4, embedDim: int = 160,
        depth: int = 4, numHeads: int = 4, patchSize: int = 40, dropout: float = 0.2,
        addPositional: bool = True
    ):
        super().__init__()
        self.numLeads = numLeads
        self.numClasses = numClasses
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.addPositional = addPositional

        # Patch embedding: [B, C, T] -> [B, C, N, D]
        self.patch = PatchEmbed1D(numLeads = numLeads, patchSize = patchSize, embedDim = embedDim)

        # Stacked CCT blocks
        self.blocks = nn.ModuleList(
            [CCTBlock(dim = embedDim, numHeads = numHeads, mlpRatio = 4.0, dropout = dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embedDim)
        self.head = nn.Linear(embedDim, numClasses)

        # Lazy positional/lead embeddings
        self.posEmbed = None   # [1, 1, N, D]
        self.leadEmbed = None  # [1, C, 1, D]

    def _maybeInitEmbeds(self, N: int, device):
        if not self.addPositional:
            return
        if self.posEmbed is None:
            self.posEmbed = nn.Parameter(torch.zeros(1, 1, N, self.embedDim, device = device))
            nn.init.trunc_normal_(self.posEmbed, std = 0.02)
        if self.leadEmbed is None:
            self.leadEmbed = nn.Parameter(torch.zeros(1, self.numLeads, 1, self.embedDim, device = device))
            nn.init.trunc_normal_(self.leadEmbed, std = 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        B, C, T = x.shape
        assert C == self.numLeads, f"Expected {self.numLeads} leads, got {C}"

        # Patchify: [B, C, T] -> [B, C, N, D]
        z = self.patch(x)
        _, C, N, D = z.shape

        # Add Positional & Lead embeddings
        if self.addPositional:
            self._maybeInitEmbeds(N, z.device)
            z = z + self.posEmbed + self.leadEmbed

        # CCT blocks
        for blk in self.blocks:
            # [B, C, N, D]
            z = blk(z)

        # Norm + Per-token head
        # [B, C, N, D]
        z = self.norm(z)
        # [B, C, N, K]
        tokenLogits = self.head(z)

        # Per-lead output
        # [B, C, N, K] -> [B, C, K, N]
        t = tokenLogits.permute(0, 1, 3, 2)
        # Collapse (C, K) to channels: [B, C * K, N]
        t = t.reshape(B, C * self.numClasses, N)
        # Upsample tokens back to per-sample: [B, C * K, T]
        t = F.interpolate(t, size = T, mode = "linear", align_corners = False)
        # Restore [B, C, K, T]
        logits = t.view(B, C, self.numClasses, T)
        return logits