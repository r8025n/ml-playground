import torch

print(torch.__version__)

def double_conv(in_channel, out_channel):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channel, out_channel, kernel_size=3),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(out_channel, out_channel, kernel_size=3),
        torch.nn.ReLU(inplace=True)
    )

class Unet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = double_conv(3, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        # self.up_conv1 = double_conv(1024, 512)
        # self.up_conv2 = double_conv()

    def forward(self, image):
        x = self.down_conv1(image)
        x = self.max_pool(x)
        x = self.down_conv2(x)
        x = self.max_pool(x)
        x = self.down_conv3(x)
        x = self.max_pool(x)
        x = self.down_conv4(x)
        x = self.max_pool(x)
        x = self.down_conv5(x)
        # x = self.max_pool(x)
        return x

if __name__ == "__main__":
    image = torch.rand((1, 3, 572, 572))  # Change input channel to 3
    model = Unet()
    print(model(image))