import matplotlib.pyplot as plt
from models import *
import torch

# List of models with their names and accuracies
model_data = [
    {'name': 'VGG16',            'net': VGG('VGG16'),           'accuracy': 92.64},
    {'name': 'ResNet18',         'net': ResNet18(),             'accuracy': 93.02},
    {'name': 'ResNet50',         'net': ResNet50(),             'accuracy': 93.62},
    {'name': 'ResNet101',        'net': ResNet101(),            'accuracy': 93.75},
    {'name': 'RegNetX_200MF',    'net': RegNetX_200MF(),        'accuracy': 94.24},
    {'name': 'RegNetX_400MF',    'net': RegNetX_400MF(),        'accuracy': 94.29},
    {'name': 'MobileNetV2',      'net': MobileNetV2(),          'accuracy': 94.43},
    {'name': 'ResNeXt29_32x4d',  'net': ResNeXt29_32x4d(),      'accuracy': 94.73},
    {'name': 'ResNeXt29_2x64d',  'net': ResNeXt29_2x64d(),      'accuracy': 94.82},
    {'name': 'SimpleDLA',        'net': SimpleDLA(),            'accuracy': 94.89},
    {'name': 'DenseNet121',      'net': DenseNet121(),          'accuracy': 95.04},
    {'name': 'PreActResNet18',   'net': PreActResNet18(),       'accuracy': 95.11},
    {'name': 'DPN92',            'net': DPN92(),                'accuracy': 95.16},
    {'name': 'DLA',              'net': DLA(),                  'accuracy': 95.47},
    {'name' : "ResNet18-adam-plateau (test)","net" : ResNet18(), "accuracy" : 90.59},
    {'name' : "ResNet18-sgd-cosine (test)","net" : ResNet18(), "accuracy" : 90.62},
    {'name' : "ResNet18-sgd-plateau (test)","net" : ResNet18(), "accuracy" : 90.15},
    {'name' : "ResNet18-adam-plateau (train)","net" : ResNet18(), "accuracy" : 94.97},
    {'name' : "ResNet18-sgd-cosine (train)","net" : ResNet18(), "accuracy" : 95.84},
    {'name' : "ResNet18-sgd-plateau (train)","net" : ResNet18(), "accuracy" : 94.66},
]

# Compute number of parameters for each model
for data in model_data:
    data['params'] = sum(p.numel() for p in data['net'].parameters())

# Plot
plt.figure(figsize=(12,6))

# Scatter points
for data in model_data:
    plt.scatter(data['params'], data['accuracy'])
    plt.annotate(data['name'], 
                 (data['params'], data['accuracy']),
                 textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

# Horizontal line at 90% accuracy
plt.axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='Accuracy = 90%')

plt.xlabel("Number of Parameters")
plt.ylabel("Accuracy (%)")
plt.title("CIFAR Model Accuracy vs Number of Parameters")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("src/accuracy_vs_params.png", dpi=300)
print("Plot saved as 'accuracy_vs_params.png'")
