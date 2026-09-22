import torch
device="cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
 with torch.autocast(device_type="cuda",dtype=torch.float16): print(torch.randn(8,8,device=device).dtype)
else: print("Run on CUDA for AMP demo")