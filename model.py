import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CaptchaModel(nn.Module):
    def __init__(self, n_classes, lstm_hidden=512, lstm_layers=2):
        super().__init__()

        # all resnet18 layers up to avgpool
        self.resnet_features = nn.Sequential(*list(resnet18(pretrained=False).children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(512, lstm_hidden, lstm_layers)
        self.conv = nn.Conv1d(lstm_hidden, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.resnet_features(x)
        batch_dim, ch_dim, h_dim, w_dim = x.shape
        x = self.pool(x)
        x = x.view(batch_dim, ch_dim, w_dim * 1)
        # shape=[b_dim, ch_dim, w_dim] -> [w_dim, b_dim, resnet_dim]
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        # shape=[w_dim, b_dim, lstm_hid] -> [b_dim, lstm_hid, w_dim]
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        # shape=[b_dim, cls_dim, w_dim] -> [w_dim, b_dim, cls_dim]
        x = x.permute(2, 0, 1)
        x = F.log_softmax(x, 2)
        return x


class CaptchaModelFixedSize(nn.Module):
    def __init__(self, n_classes, output_width=5):
        super().__init__()

        self.n_classes = n_classes
        self.output_width = output_width
        # all resnet18 layers up to avgpool
        self.resnet_ch = 512
        self.resnet_features = nn.Sequential(*list(resnet18(pretrained=False).children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, self.output_width))
        self.linear = nn.Linear(self.output_width * self.resnet_ch, self.output_width * self.n_classes)

    def forward(self, x):
        x = self.resnet_features(x)
        x = self.pool(x)
        batch_dim = x.shape[0]
        x = x.view(batch_dim, -1)
        x = self.linear(x)
        x = x.view(batch_dim, self.n_classes, self.output_width)
        x = F.log_softmax(x, 1)
        return x


if __name__ == '__main__':
    batch = 1
    channels = 3
    h = 150
    w = 330
    classes = 30 + 1
    label_size = 5

    in_tensor = torch.rand(size=(batch, channels, h, w), dtype=torch.float)
    targets = torch.randint(low=1, high=classes, size=(batch, label_size), dtype=torch.long)
    target_lens = torch.full(size=(batch,), fill_value=label_size, dtype=torch.long)

    # model = CaptchaModel(classes)
    model = CaptchaModelFixedSize(classes, label_size)
    print('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters()
                                                          if p.requires_grad is True)))
    print(model)
    # criterion = nn.CTCLoss()
    criterion = nn.NLLLoss()

    output = model(in_tensor)
    print(output.shape)
    input_lens = torch.full(size=(batch,), fill_value=output.shape[0], dtype=torch.long)
    # loss = criterion(output, targets, input_lens, target_lens)
    loss = criterion(output, targets)
    loss.backward()
