import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms

from PIL import Image


class ResNet50Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        # ── LOAD PRETRAINED RESNET50 ──
        # weights='IMAGENET1K_V1' means load weights trained
        # on ImageNet. These weights encode 50 layers worth of
        # visual knowledge from 1.2 million images.
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # ── REMOVE LAST TWO LAYERS ──
        # ResNet50 ends with: avgpool → flatten → fc (classifier)
        # We don't want the classifier — we want the features.
        # list(resnet.children())[:-2] removes avgpool and fc.
        # What remains outputs shape: (batch, 2048, 7, 7)
        # 2048 channels of deeply learned features at 7x7 spatial
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ── FREEZE BACKBONE WEIGHTS ──
        # WHY FREEZE? We don't want to change ResNet50's weights
        # during training. They are already excellent.
        # Freezing means only our new layers (reduce, upsample,
        # CSPNet, classifier) learn during training.
        # This also speeds up training significantly.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── CHANNEL REDUCTION ──
        # ResNet50 outputs 2048 channels — too many for our
        # lightweight CSPNet. Reduce to 256 channels using
        # a 1x1 conv (channel mixing without spatial change).
        # WHY 256? Rich enough for CSPNet, light enough to train.
        # 1x1 conv: (batch, 2048, 7, 7) → (batch, 256, 7, 7)
        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # ── UPSAMPLE TO 56x56 ──
        # WHY UPSAMPLE? Our patchify cuts 56x56 into
        # 4x4 = 16 patches of 14x14 each.
        # ResNet50 gives us 7x7 — too small for that.
        # Bilinear upsampling stretches 7x7 → 56x56
        # without any learnable parameters.
        # mode='bilinear': smooth interpolation between pixels
        # align_corners=False: standard setting for bilinear
        self.upsample = nn.Upsample(
            size=(56, 56),
            mode='bilinear',
            align_corners=False
        )

    def forward(self, x):
        # x: (batch, 3, 224, 224)

        x = self.backbone(x)   # → (batch, 2048, 7, 7)
        x = self.reduce(x)     # → (batch, 256, 7, 7)
        x = self.upsample(x)   # → (batch, 256, 56, 56)

        return x
        # Output: rich 256-channel feature map at 56x56
        # Ready for patchify — same as before

def patchify(feature_map, patch_size=14):
    """
    Cuts feature map into equal patches.
    IDENTICAL to before — nothing changed here.
    56x56 ÷ 14 = 4 patches per side = 16 patches total
    """
    batch_size, channels, height, width = feature_map.shape
    num_patches_h = height // patch_size
    num_patches_w = width  // patch_size
    patches = []

    for row in range(num_patches_h):
        for col in range(num_patches_w):
            r_start = row * patch_size
            c_start = col * patch_size
            patch   = feature_map[
                :, :,
                r_start:r_start+patch_size,
                c_start:c_start+patch_size
            ]
            patches.append(patch)

    return torch.stack(patches, dim=1)
    # Output: (batch, 16, 256, 14, 14)

class FeatureExtractorAndPatcher(nn.Module):
    """
    Chains ResNet50 backbone + patchify.
    Replaces the old FeatureExtractorAndPatcher_B.
    """

    def __init__(self, patch_size=14):
        super().__init__()
        self.backbone   = ResNet50Backbone()
        self.patch_size = patch_size

    def forward(self, x):
        # Step 1: ResNet50 extracts rich features
        # (batch, 3, 224, 224) → (batch, 256, 56, 56)
        x = self.backbone(x)

        # Step 2: Patchify — identical to before
        # (batch, 256, 56, 56) → (batch, 16, 256, 14, 14)
        patches = patchify(x, patch_size=self.patch_size)

        return patches
class CSPBlock(nn.Module):
    """
    Cross Stage Partial Network block.
    IDENTICAL to before — no changes.
    Splits input, processes half, merges back.
    """

    def __init__(self, in_channels):
        super().__init__()
        half = in_channels // 2

        self.main_conv1 = nn.Conv2d(half, half, kernel_size=3, padding=1)
        self.main_bn1   = nn.BatchNorm2d(half)
        self.main_relu1 = nn.ReLU()

        self.main_conv2 = nn.Conv2d(half, half, kernel_size=3, padding=1)
        self.main_bn2   = nn.BatchNorm2d(half)
        self.main_relu2 = nn.ReLU()

        self.blend_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.blend_bn   = nn.BatchNorm2d(in_channels)
        self.blend_relu = nn.ReLU()

    def forward(self, x):
        main_input, skip_input = torch.chunk(x, chunks=2, dim=1)

        main = self.main_relu1(self.main_bn1(self.main_conv1(main_input)))
        main = self.main_relu2(self.main_bn2(self.main_conv2(main)))

        combined = torch.cat([main, skip_input], dim=1)
        out      = self.blend_relu(self.blend_bn(self.blend_conv(combined)))

        return out

class PatchCSPNet(nn.Module):
    """
    Processes all 16 patches through shared CSPBlock.
    Aggregates patch features by averaging.
    IDENTICAL to before — only in_channels changes 128→256
    """

    def __init__(self, in_channels=256):
        super().__init__()
        self.csp_block = CSPBlock(in_channels=in_channels)
        self.gap        = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, patches):
        # patches: (batch, 16, 256, 14, 14)
        batch_size  = patches.shape[0]
        in_channels = patches.shape[2]  # 256
        patch_features = []

        for i in range(patches.shape[1]):  # loop 16 patches
            patch    = patches[:, i, :, :, :]           # (batch, 256, 14, 14)
            features = self.csp_block(patch)             # (batch, 256, 14, 14)
            features = self.gap(features)                # (batch, 256, 1, 1)
            features = features.view(batch_size, in_channels)  # (batch, 256)
            patch_features.append(features)

        all_features  = torch.stack(patch_features, dim=1)  # (batch, 16, 256)
        image_feature = all_features.mean(dim=1)             # (batch, 256)

        return image_feature

class ClassificationHead(nn.Module):
    """
    Maps 256-dim feature vector to 6 AQI class scores.
    IDENTICAL to before — only in_features changes 128→256
    """

    def __init__(self, in_features=256, num_classes=6):
        super().__init__()
        self.fc1     = nn.Linear(in_features, 512)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc2     = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AQIClassifier(nn.Module):
    """
    Complete pipeline:
    Raw image → ResNet50 → Patchify → CSPNet → Classify
    """

    def __init__(self, num_classes=6, patch_size=14):
        super().__init__()

        # ResNet50 replaces StridedConvBlock
        self.feature_extractor = FeatureExtractorAndPatcher(
            patch_size=patch_size
        )
        # CSPNet — identical to before, just 256 channels
        self.patch_cspnet = PatchCSPNet(in_channels=256)

        # Classifier — identical to before, just 256 input
        self.classifier = ClassificationHead(
            in_features=256,
            num_classes=num_classes
        )

    def forward(self, x):
        patches        = self.feature_extractor(x)  # (batch, 16, 256, 14, 14)
        image_features = self.patch_cspnet(patches)  # (batch, 256)
        logits         = self.classifier(image_features)  # (batch, 6)
        return logits
