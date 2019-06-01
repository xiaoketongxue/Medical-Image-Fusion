from unet_parts import *

class FWNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FWNet, self).__init__()
        
        self.inc_gen = inconv(2, 32)
        self.down1_gen = down(32, 64)
        self.down2_gen = down(64, 128)
        self.down3_gen = down(128, 256)
        self.down4_gen = down(256, 256)
        self.up1_gen = up(512, 128)
        self.up2_gen = up(256, 64)
        self.up3_gen = up(128, 32)
        self.up4_gen = up(64, 32)
        self.outc_gen = outconv(32, 1)
        
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)
        
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        # generate network
        x1 = self.inc_gen(x)
        x2 = self.down1_gen(x1)
        x3 = self.down2_gen(x2)
        x4 = self.down3_gen(x3)
        x5 = self.down4_gen(x4)
        x = self.up1_gen(x5, x4)
        x = self.up2_gen(x, x3)
        x = self.up3_gen(x, x2)
        x = self.up4_gen(x, x1)
        x = self.outc_gen(x)
        fr = F.sigmoid(x)
        
        # rebuild network
        x1 = self.inc(fr)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return fr, F.sigmoid(x)
