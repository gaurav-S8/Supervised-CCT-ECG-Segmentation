import torch
from models import UNet1D, CNNBiLSTM_Model_Profile11

def testForwardPass_UNet1D():
    model = UNet1D(input_channels = 1, num_classes = 4)
    # batch = 2, 1 lead, 2000 samples (4s @ 500Hz)
    x = torch.randn(2, 1, 2000)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 4, 2000), "Output shape mismatch!"
    print("For 1D U-Net model! Forward pass test passed!")

def testForwardPass_CNNBiLSTM():
    model = CNNBiLSTM_Model_Profile11(input_channels = 1, num_classes = 4)
    # batch = 2, 1 lead, 2000 samples (4s @ 500Hz)
    x = torch.randn(2, 1, 2000)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 4, 2000), "Output shape mismatch!"
    print("For CNN+BiLSTM hybrid model! Forward pass test passed!")