# Model.py
import torch
import torch.nn as nn
from Module import (
    StarNet, Conv, CAFE, AFFM, EUCB, CMFF, AMFB, RTDETRDecoder
)

__all__ = ['STCNet']

# ----------------------------
# STCNet Model
# ----------------------------
class STCNet(nn.Module):
    def __init__(self, nc=80, head_channel=128):
        super().__init__()
        self.nc = nc
        self.head_channel = head_channel
        
        # Backbone
        self.backbone = StarNet(base_dim=16, depths=(1, 1, 3, 1))
        
        # Head
        self.conv1 = Conv(256, 256, 1, 1, None, 1, 1, False)  # 5
        self.cafe = CAFE(256, 1024, 8)  # 6 (formerly TransformerEncoderLayer_AdditiveTokenMixer)
        self.conv2 = Conv(256, 256, 1, 1)  # 7
        
        # Feature processing
        self.conv_p2 = Conv(16, head_channel, 1, 1)  # 8 (from P2)
        self.conv_p3 = Conv(32, head_channel, 1, 1)  # 9 (from P3)
        self.conv_p4 = Conv(256, head_channel, 1, 1)  # 10 (from P4)
        
        # Neck layers
        self.conv11 = Conv(head_channel, head_channel, 3, 2)  # 11
        self.amfb1 = AMFB(head_channel, [5, 7, 9])  # 12-13 (AFFM + 3x CMFF)
        
        self.eucb1 = EUCB(head_channel)  # 14
        self.conv15 = Conv(head_channel, head_channel, 3, 2)  # 15
        self.amfb2 = AMFB(head_channel, [3, 5, 7])  # 16-17 (AFFM + 3x CMFF)
        
        self.eucb2 = EUCB(head_channel)  # 18
        self.conv19 = Conv(head_channel, head_channel, 3, 2)  # 19
        self.amfb3 = AMFB(head_channel, [1, 3, 5])  # 20-21 (AFFM + 3x CMFF)
        
        self.affm1 = AFFM([head_channel, head_channel])  # 22
        self.cmff1 = CMFF(head_channel, head_channel, 3, [1, 3, 5])  # 23
        
        self.conv24 = Conv(head_channel, head_channel, 3, 2)  # 24
        self.conv25 = Conv(head_channel, head_channel, 3, 2)  # 25
        self.affm2 = AFFM([head_channel, head_channel, head_channel, head_channel])  # 26
        self.cmff2 = CMFF(head_channel, head_channel, 3, [3, 5, 7])  # 27
        
        self.conv28 = Conv(head_channel, head_channel, 3, 2)  # 28
        self.conv29 = Conv(head_channel, head_channel, 3, 2)  # 29
        self.affm3 = AFFM([head_channel, head_channel, head_channel])  # 30
        self.cmff3 = CMFF(head_channel, head_channel, 3, [5, 7, 9])  # 31
        
        # Detection head
        self.detector = RTDETRDecoder(nc, 256, 300, 4, 8)  # 32

    def forward(self, x):
        # Backbone
        backbone_features = self.backbone(x)
        p2, p3, p4, p5 = backbone_features  # P2, P3, P4, P5
        
        # Head
        x = self.conv1(p5)
        x = self.cafe(x)
        x7 = self.conv2(x)  # 7
        
        # Process features
        x8 = self.conv_p2(p2)  # 8
        x9 = self.conv_p3(p3)  # 9
        x10 = self.conv_p4(p4)  # 10
        
        # Neck
        x11 = self.conv11(x9)  # 11
        x13 = self.amfb1([x11, x10])  # 12-13 (AFFM + 3x CMFF)
        
        x14 = self.eucb1(x13)  # 14
        x15 = self.conv15(x8)  # 15
        x17 = self.amfb2([x15, x14, x9])  # 16-17 (AFFM + 3x CMFF)
        
        x18 = self.eucb2(x17)  # 18
        x19 = self.conv19(x15)  # 19
        x21 = self.amfb3([x19, x18, x8])  # 20-21 (AFFM + 3x CMFF)
        
        x22 = self.affm1([x18, x21])  # 22
        x23 = self.cmff1(x22)  # 23
        
        x24 = self.conv24(x21)  # 24
        x25 = self.conv25(x23)  # 25
        x26 = self.affm2([x25, x24, x17, x14])  # 26
        x27 = self.cmff2(x26)  # 27
        
        x28 = self.conv28(x17)  # 28
        x29 = self.conv29(x27)  # 29
        x30 = self.affm3([x29, x28, x13])  # 30
        x31 = self.cmff3(x30)  # 31
        
        # Detection
        outputs = self.detector([x23, x27, x31])
        return outputs


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Create STCNet model
    model = STCNet(nc=80, head_channel=128)
    
    # Test the model
    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    print("STCNet output shapes:")
    print(f"  pred_logits: {out['pred_logits'].shape}")
    print(f"  pred_boxes: {out['pred_boxes'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
