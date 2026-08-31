from torchvision import transforms
transform=transforms.Compose([transforms.Resize((128,128)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
print(transform)
