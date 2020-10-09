import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CaptchaModel(nn.Module):
    def __init__(self, in_channels, n_classes, height):
        super().__init__()

        # self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet = nn.Sequential(*list(resnet18(pretrained=False).children())[:-4])  # up to layer 2 / 4
        print(self.resnet)
        resnet_dim = 128 * (height // 8 + 1)
        self.gru = nn.GRU(resnet_dim, n_classes)  #, num_layers=2, bidirectional=True)
        # self.fc = nn.Linear(resnet_dim, n_classes)

    def forward(self, x):
        # x = self.conv(x)
        # x = self.max_pool(x)
        # x = F.relu(x)
        x = self.resnet(x)

        batch_size = x.shape[0]
        width = x.shape[3]
        # shape=[batch_dim, ch_dim, h_dim, w_dim] -> [w_dim, b_dim, ch_dim x h_dim]
        x = x.permute(3, 0, 1, 2)
        x = x.view(width, batch_size, -1)
        x, _ = self.gru(x)
        x = F.log_softmax(x, 2)
        return x


if __name__ == '__main__':
    batch = 1
    channels = 3
    h = 150
    w = 330
    classes = 30 + 1
    label_size = 5

    in_tensor = torch.rand(size=(batch, channels, h, w), dtype=torch.float)
    targets = [torch.randint(low=1, high=classes, size=(label_size,), dtype=torch.int32) for _ in range(batch)]
    targets = torch.stack(targets)
    target_lens = torch.full(size=(batch,), fill_value=label_size, dtype=torch.int32)

    model = CaptchaModel(channels, classes, h)
    print('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters()
                                                          if p.requires_grad is True)))
    print(model)
    criterion = nn.CTCLoss()

    output = model(in_tensor)
    input_lens = torch.full(size=(batch,), fill_value=output.shape[0], dtype=torch.int32)
    loss = criterion(output, targets, input_lens, target_lens)
    loss.backward()
