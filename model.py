import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptchModel(nn.Module):
    def __init__(self, num_char):
        super(CaptchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(1152, 64)
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.GRU(64, 32, bidirectional=True, dropout=0.25, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(64, num_char+1)


    def forward(self, images, targets=None):
        bth, c, h, w = images.size()
        # print(bth, c, h, w)
        x = F.relu(self.conv1(images))
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())
        x = F.relu(self.conv2(x))
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        x = x.permute(0, 3, 1, 2)
        # print(x.size())
        x = x.view(bth, x.size(1), -1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x, _ = self.lstm(x)
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        x = x.permute(1, 0, 2)
        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bth,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            # print(input_lengths)
            target_lengths = torch.full(
                size=(bth,), fill_value=targets.size(1), dtype=torch.int32
            )
            # print(target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None

if __name__ == "__main__":
    cm = CaptchModel(19)
    images = torch.randn(5, 3, 75, 300)
    targets = torch.randint(1, 20, (5, 5))
    x, loss = cm(images, targets)


