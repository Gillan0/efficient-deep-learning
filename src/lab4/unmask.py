import torch
import torch.nn.utils.prune as prune
from resnet import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the masked pruned model
checkpoint_path = "./src/lab4/checkpoint/adam-mixup_pruned_retrained_60_95"  # change to your file
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# 1️⃣ Recreate model
net = ResNet18().to(device)

# 2️⃣ IMPORTANT: Apply dummy pruning so weight_orig / weight_mask exist
for module in net.modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        prune.identity(module, name="weight")  
        # identity creates weight_orig + weight_mask without changing weights

# 3️⃣ Now load masked state_dict
net.load_state_dict(checkpoint['net'])

# 4️⃣ Permanently apply pruning
for module in net.modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        try:
            prune.remove(module, 'weight')
        except ValueError:
            pass

# 5️⃣ Save clean model
save_path = checkpoint_path + "_final"
torch.save({
    'net': net.state_dict(),
    'acc': checkpoint['acc'],
    'epoch': checkpoint['epoch'],
}, save_path)

print(f"Saved unmasked model to {save_path}")