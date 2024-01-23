import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv_block = nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU(inplace=True)
                          )
    
    def forward(self, x):
        
        output = self.conv_block(x)
        return output


class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        
        self.down_conv = nn.Sequential(
                         nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True)
                         )
    
    def forward(self, x):
        
        output = self.down_conv(x)
        return output


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        
        self.up_conv = nn.Sequential(
                       nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                       nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace=True)
                       )
    
    def forward(self, x):
        
        output = self.up_conv(x)
        return output


class inter_conv(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(inter_conv, self).__init__()
        
        self.inter_conv = nn.Sequential(
                          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                          nn.BatchNorm2d(out_ch),
                          nn.ReLU(inplace=True),
                          nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                          )
    
    def forward(self, x):
        
        output = self.inter_conv(x)
        return output


class UNetSharpSharp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetSharpSharp, self).__init__()
        
        filters=[64, 128, 256, 512, 1024]
        
        # Step-1: Encode-0 Blocks of the U-Nets
        self.Input_to_B01234_0   = conv_block(in_ch, filters[0])
        
        # Step-2: Encode-1 Blocks of the U-Nets
        self.B01234_0_to_B0123_1 = conv_block(filters[0], filters[0])
        self.B01234_0_to_B4_1    = down_conv (filters[0], filters[1])
        
        # Step-3: Auxiliary Convolution Blocks for the Encode-1 Feature Maps
        self.B0123_1_to_B4_1     = inter_conv(filters[0], filters[1], 0.5)
        
        # Step-4: Encode-2 Blocks of the U-Nets
        self.B0123_1_to_B012_2   = conv_block(filters[0], filters[0])
        self.B0123_1_to_B3_2     = down_conv (filters[0], filters[1])
        self.B4_1_to_B4_2        = down_conv (filters[1], filters[2])
        
        # Step-5: Auxiliary Convolution Blocks for the Encode-2 Feature Maps
        self.B012_2_to_B3_2      = inter_conv(filters[0], filters[1], 0.5)
        
        self.B012_2_to_B4_2      = inter_conv(filters[0], filters[2], 0.25)
        self.B3_2_to_B4_2        = inter_conv(filters[1], filters[2], 0.5)
        
        # Step-6: Encode-3 Blocks of the U-Nets
        self.B012_2_to_B01_3     = conv_block(filters[0], filters[0])
        self.B012_2_to_B2_3      = down_conv (filters[0], filters[1])
        self.B3_2_to_B3_3        = down_conv (filters[1], filters[2])
        self.B4_2_to_B4_3        = down_conv (filters[2], filters[3])
        
        # Step-7: Auxiliary Convolution Blocks for the Encode-3 Feature Maps
        self.B01_3_to_B2_3       = inter_conv(filters[0], filters[1], 0.5)
        
        self.B01_3_to_B3_3       = inter_conv(filters[0], filters[2], 0.25)
        self.B2_3_to_B3_3        = inter_conv(filters[1], filters[2], 0.5)
        
        self.B01_3_to_B4_3       = inter_conv(filters[0], filters[3], 0.125)
        self.B2_3_to_B4_3        = inter_conv(filters[1], filters[3], 0.25)
        self.B3_3_to_B4_3        = inter_conv(filters[2], filters[3], 0.5)
        
        # Step-8: Encode-4 Blocks of the U-Nets
        self.B01_3_to_B0_4       = conv_block(filters[0], filters[0])
        self.B01_3_to_B1_4       = down_conv (filters[0], filters[1])
        self.B2_3_to_B2_4        = down_conv (filters[1], filters[2])
        self.B3_3_to_B3_4        = down_conv (filters[2], filters[3])
        self.B4_3_to_B4_4        = down_conv (filters[3], filters[4])
        
        # Step-9: Auxiliary Convolution Blocks for the Encode-4 Feature Maps
        self.B0_4_to_B1_4        = inter_conv(filters[0], filters[1], 0.5)
        
        self.B0_4_to_B2_4        = inter_conv(filters[0], filters[2], 0.25)
        self.B1_4_to_B2_4        = inter_conv(filters[1], filters[2], 0.5)
        
        self.B0_4_to_B3_4        = inter_conv(filters[0], filters[3], 0.125)
        self.B1_4_to_B3_4        = inter_conv(filters[1], filters[3], 0.25)
        self.B2_4_to_B3_4        = inter_conv(filters[2], filters[3], 0.5)
        
        self.B0_4_to_B4_4        = inter_conv(filters[0], filters[4], 0.0625)
        self.B1_4_to_B4_4        = inter_conv(filters[1], filters[4], 0.125)
        self.B2_4_to_B4_4        = inter_conv(filters[2], filters[4], 0.25)
        self.B3_4_to_B4_4        = inter_conv(filters[3], filters[4], 0.5)
        
        # Step-10: Decode-3 Blocks of the U-Nets
        self.B0_4_to_B01_5       = conv_block(filters[0], filters[0])
        self.B1_4_to_B01_5       = up_conv   (filters[1], filters[0])
        self.B01_5_Cat_Blocks    = conv_block(2 * filters[0], filters[0])
        self.B2_4_to_B2_5        = up_conv   (filters[2], filters[1])
        self.B3_4_to_B3_5        = up_conv   (filters[3], filters[2])
        self.B4_4_to_B4_5        = up_conv   (filters[4], filters[3])
        
        # Step-11: Auxiliary Convolution Blocks for the Decode-3 Feature Maps
        self.B01_5_to_B2_5       = inter_conv(filters[0], filters[1], 0.5)
        
        self.B01_5_to_B3_5       = inter_conv(filters[0], filters[2], 0.25)
        self.B2_5_to_B3_5        = inter_conv(filters[1], filters[2], 0.5)
        
        self.B01_5_to_B4_5       = inter_conv(filters[0], filters[3], 0.125)
        self.B2_5_to_B4_5        = inter_conv(filters[1], filters[3], 0.25)
        self.B3_5_to_B4_5        = inter_conv(filters[2], filters[3], 0.5)
        
        # Step-12: Convolution Blocks for the Skip Connection Concatenations on the Decode-3 Blocks
        self.B01_5_Cat_Skips     = conv_block(2 * filters[0], filters[0])
        self.B2_5_Cat_Skips      = conv_block(2 * filters[1], filters[1])
        self.B3_5_Cat_Skips      = conv_block(2 * filters[2], filters[2])
        self.B4_5_Cat_Skips      = conv_block(2 * filters[3], filters[3])
        
        # Step-13: Decode-2 Blocks of the U-Nets
        self.B01_5_to_B012_6     = conv_block(filters[0], filters[0])
        self.B2_5_to_B012_6      = up_conv   (filters[1], filters[0])
        self.B012_6_Cat_Blocks   = conv_block(2 * filters[0], filters[0])
        self.B3_5_to_B3_6        = up_conv   (filters[2], filters[1])
        self.B4_5_to_B4_6        = up_conv   (filters[3], filters[2])
        
        # Step-14: Auxiliary Convolution Blocks for the Decode-2 Feature Maps
        self.B012_6_to_B3_6      = inter_conv(filters[0], filters[1], 0.5)
        
        self.B012_6_to_B4_6      = inter_conv(filters[0], filters[2], 0.25)
        self.B3_6_to_B4_6        = inter_conv(filters[1], filters[2], 0.5)
        
        # Step-15: Convolution Blocks for the Skip Connection Concatenations on the Decode-2 Blocks
        self.B012_6_Cat_Skips    = conv_block(2 * filters[0], filters[0])
        self.B3_6_Cat_Skips      = conv_block(2 * filters[1], filters[1])
        self.B4_6_Cat_Skips      = conv_block(2 * filters[2], filters[2])
        
        # Step-16: Decode-1 Blocks of the U-Nets
        self.B012_6_to_B0123_7   = conv_block(filters[0], filters[0])
        self.B3_6_to_B0123_7     = up_conv   (filters[1], filters[0])
        self.B0123_7_Cat_Blocks  = conv_block(2 * filters[0], filters[0])
        self.B4_6_to_B4_7        = up_conv   (filters[2], filters[1])
        
        # Step-17: Auxiliary Convolution Blocks for the Decode-1 Feature Maps
        self.B0123_7_to_B4_7     = inter_conv(filters[0], filters[1], 0.5)
        
        # Step-18: Convolution Blocks for the Skip Connection Concatenations on the Decode-1 Blocks
        self.B0123_7_Cat_Skips   = conv_block(2 * filters[0], filters[0])
        self.B4_7_Cat_Skips      = conv_block(2 * filters[1], filters[1])
        
        # Step:19: Decode-0 Blocks of the U-Nets
        self.B0123_7_to_B01234_8 = conv_block(filters[0], filters[0])
        self.B4_7_to_B01234_8    = up_conv   (filters[1], filters[0])
        self.B01234_8_Cat_Blocks = conv_block(2 * filters[0], filters[0])
        
        # Step-20: Convolution Blocks for the Skip Connection Concatenations on the Decode-0 Blocks
        self.B01234_8_Cat_Skips  = conv_block(2 * filters[0], filters[0])
        
        # Step-21: Deeply Supervised Output Blocks
        self.final_d0 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.final_d1 = nn.Sequential(
                        nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.final_d2 = nn.Sequential(
                        nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        
        self.final_d3 = nn.Sequential(
                        nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        
        self.final_d4 = nn.Sequential(
                        nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
    
    
    def forward(self, x):
        
        # Step-1: Encode-0 Blocks of the U-Nets
        x01234_0 = self.Input_to_B01234_0(x)
        
        # Step-2: Encode-1 Blocks of the U-Nets
        x0123_1  = self.B01234_0_to_B0123_1(x01234_0)
        x4_1     = self.B01234_0_to_B4_1   (x01234_0)
        
        # Step-3: Addition of the Encode-1 Feature Maps
        x4_1     = ( x4_1 + self.B0123_1_to_B4_1(x0123_1) )
        
        # Step-4: Encode-2 Blocks of the U-Nets
        x012_2   = self.B0123_1_to_B012_2(x0123_1)
        x3_2     = self.B0123_1_to_B3_2  (x0123_1)
        x4_2     = self.B4_1_to_B4_2     (x4_1)
        
        # Step-5: Addition of the Encode-2 Feature Maps
        x4_2     = ( x4_2 + self.B012_2_to_B4_2(x012_2) + self.B3_2_to_B4_2(x3_2) )
        x3_2     = ( x3_2 + self.B012_2_to_B3_2(x012_2) )
        
        # Step-6: Encode-3 Blocks of the U-Nets
        x01_3    = self.B012_2_to_B01_3(x012_2)
        x2_3     = self.B012_2_to_B2_3 (x012_2)
        x3_3     = self.B3_2_to_B3_3   (x3_2)
        x4_3     = self.B4_2_to_B4_3   (x4_2)
        
        # Step-7: Addition of the Encode-3 Feature Maps
        x4_3     = ( x4_3  + self.B01_3_to_B4_3(x01_3) + self.B2_3_to_B4_3(x2_3) + self.B3_3_to_B4_3(x3_3) )
        x3_3     = ( x3_3  + self.B01_3_to_B3_3(x01_3) + self.B2_3_to_B3_3(x2_3) )
        x2_3     = ( x2_3  + self.B01_3_to_B2_3(x01_3) )
        
        # Step-8: Encode-4 Blocks of the U-Nets
        x0_4     = self.B01_3_to_B0_4(x01_3)
        x1_4     = self.B01_3_to_B1_4(x01_3)
        x2_4     = self.B2_3_to_B2_4 (x2_3)
        x3_4     = self.B3_3_to_B3_4 (x3_3)
        x4_4     = self.B4_3_to_B4_4 (x4_3)
        
        # Step-9: Addition of the Encode-4 Feature Maps
        x4_4     = ( x4_4 + self.B0_4_to_B4_4(x0_4) + self.B1_4_to_B4_4(x1_4) + self.B2_4_to_B4_4(x2_4) + self.B3_4_to_B4_4(x3_4) )
        x3_4     = ( x3_4 + self.B0_4_to_B3_4(x0_4) + self.B1_4_to_B3_4(x1_4) + self.B2_4_to_B3_4(x2_4) )
        x2_4     = ( x2_4 + self.B0_4_to_B2_4(x0_4) + self.B1_4_to_B2_4(x1_4) )
        x1_4     = ( x1_4 + self.B0_4_to_B1_4(x0_4) )
        
        # Step-10: Decode-3 Blocks of the U-Nets
        x01_5    = torch.cat((self.B0_4_to_B01_5(x0_4), self.B1_4_to_B01_5(x1_4)), dim=1)
        x01_5    = self.B01_5_Cat_Blocks(x01_5)
        x2_5     = self.B2_4_to_B2_5    (x2_4)
        x3_5     = self.B3_4_to_B3_5    (x3_4)
        x4_5     = self.B4_4_to_B4_5    (x4_4)
        
        # Step-11: Addition of the Decode-3 Feature Maps
        x4_5     = ( x4_5 + self.B01_5_to_B4_5(x01_5) + self.B2_5_to_B4_5(x2_5) + self.B3_5_to_B4_5(x3_5) )
        x3_5     = ( x3_5 + self.B01_5_to_B3_5(x01_5) + self.B2_5_to_B3_5(x2_5) )
        x2_5     = ( x2_5 + self.B01_5_to_B2_5(x01_5) )
        
        # Step-12: Concatenating Skip Connections to the Decode-3 Blocks
        x01_5    = torch.cat((x01_5, x01_3), dim=1)
        x01_5    = self.B01_5_Cat_Skips(x01_5)
        
        x2_5     = torch.cat((x2_5, x2_3), dim=1)
        x2_5     = self.B2_5_Cat_Skips(x2_5)
        
        x3_5     = torch.cat((x3_5, x3_3), dim=1)
        x3_5     = self.B3_5_Cat_Skips(x3_5)
        
        x4_5     = torch.cat((x4_5, x4_3), dim=1)
        x4_5     = self.B4_5_Cat_Skips(x4_5)
        
        # Step-13: Decode-2 Blocks of the U-Nets
        x012_6   = torch.cat((self.B01_5_to_B012_6(x01_5), self.B2_5_to_B012_6(x2_5)), dim=1)
        x012_6   = self.B012_6_Cat_Blocks(x012_6)
        x3_6     = self.B3_5_to_B3_6     (x3_5)
        x4_6     = self.B4_5_to_B4_6     (x4_5)
        
        # Step-14: Addition of the Decode-2 Feature Maps
        x4_6     = ( x4_6 + self.B012_6_to_B4_6(x012_6) + self.B3_6_to_B4_6(x3_6) )
        x3_6     = ( x3_6 + self.B012_6_to_B3_6(x012_6) )
        
        # Step-15: Concatenating Skip Connections to the Decode-2 Blocks
        x012_6   = torch.cat((x012_6, x012_2), dim=1)
        x012_6   = self.B012_6_Cat_Skips(x012_6)
        
        x3_6     = torch.cat((x3_6, x3_2), dim=1)
        x3_6     = self.B3_6_Cat_Skips(x3_6)
        
        x4_6     = torch.cat((x4_6, x4_2), dim=1)
        x4_6     = self.B4_6_Cat_Skips(x4_6)
        
        # Step-16: Decode-1 Blocks of the U-Nets        
        x0123_7  = torch.cat((self.B012_6_to_B0123_7(x012_6), self.B3_6_to_B0123_7(x3_6)), dim=1)
        x0123_7  = self.B0123_7_Cat_Blocks(x0123_7)
        x4_7     = self.B4_6_to_B4_7      (x4_6)
        
        # Step-17: Addition of the Decode-1 Feature Maps
        x4_7     = ( x4_7 + self.B0123_7_to_B4_7(x0123_7) )
        
        # Step-18: Concatenating Skip Connections to the Decode-1 Blocks
        x0123_7  = torch.cat((x0123_7, x0123_1), dim=1)
        x0123_7  = self.B0123_7_Cat_Skips(x0123_7)
        
        x4_7     = torch.cat((x4_7, x4_1), dim=1)
        x4_7     = self.B4_7_Cat_Skips(x4_7)
        
        # Step-19: Decode-0 Blocks of the U-Nets
        x01234_8 = torch.cat((self.B0123_7_to_B01234_8(x0123_7), self.B4_7_to_B01234_8(x4_7)), dim=1)
        x01234_8 = self.B01234_8_Cat_Blocks(x01234_8)
        
        # Step-20: Concatenating Skip Connections to the Decode-0 Blocks
        x01234_8 = torch.cat((x01234_8, x01234_0), dim=1)
        x01234_8 = self.B01234_8_Cat_Skips(x01234_8)
        
        # Step-21: Deeply Supervised Output Blocks
        d0 = self.final_d0(x01234_8)
        d1 = self.final_d1(x4_7)
        d2 = self.final_d2(x4_6)
        d3 = self.final_d3(x4_5)
        d4 = self.final_d4(x4_4)
        
        # Step-22: Final Output
        output = ( d0 + d1 + d2 + d3 + d4 ) / 5
        
        # Step-23: Activation Function
        #output = nn.Softmax(dim=1)(output)
        output = nn.Sigmoid()(output)
        
        #############
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSharpSharp(3, 1).to(device)
    model.eval()
    input = torch.randn(1, 3, 32, 32).to(device)
    output = model(input)
    print("Test Input :", input.size())
    print("Test Output:", output.size())