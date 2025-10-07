import torch.nn as nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SEBlock, self).__init__()

        # Squeeze: Global average pooling (done in forward)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two FC layers with bottleneck
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x)

        y = y.view(batch_size, channels)

        y = self.fc(y)

        y = y.view(batch_size, channels, 1, 1)

        return x * y.expand_as(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, use_se = True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE Block
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)
        

        # Skip Connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)
        
        out += identity
        out = self.relu(out)

        return out
    
class SEResNet(nn.Module):
    def __init__(self, num_classes = 10, use_se = True):
        super(SEResNet, self).__init__()

        self.use_se = use_se

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier 
        self.fc = nn.Linear(512, num_classes)

        # Initialise weights
        self._initialise_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        layers.append(ResidualBlock(in_channels, out_channels, stride, self.use_se))

        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, 1, self.use_se))

        return nn.Sequential(*layers)
    
    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
    
def create_se_resnet(num_classes = 10, use_se = True):
    model = SEResNet(num_classes=num_classes, use_se=use_se)
    return model