from torchvision import transforms
print(transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor()]))