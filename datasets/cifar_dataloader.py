from torchvision import transforms
from torchvision.datasets import CIFAR100
import torch

class CIFAR100_Cls(object):
    def __init__(self, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = 100

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        # 训练集
        trainset = CIFAR100(root=dataroot, train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        # 测试集
        testset = CIFAR100(root=dataroot, train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset))
