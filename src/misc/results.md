cosine-mixup-60-83 : score = 2.3250, acc = 91.520%
cosine-mixup-60-83 (half) : score = 1.1625, acc = 91.450%


adam-mixup-60-95 : score = 2.0859, Acc = 90.81
adam-mixup-60-95 (half): score = 1.0429, Acc = 89.770%

cosine-mixup : score = 3.9795, acc : 91.830%
adam-mixup : score = 3.9795, acc : 94.310%

cosine-mixup (half) : score = 1.9897, acc : 91.830%
adam-mixup (half) : score = 1.9897, acc : 94.260

mobileNet-adam-mixup (half) : score = 0.3678, acc = 92.120%
mobileNet-cosine-mixup (half) : score = 0.3678, acc = 90.620%

customNet-adam-mixup (half) : score = 1.1934, acc = 93.700%
customNet-cosine-mixup (half) : score = 1.1934, acc = 89.750%

Batch : 32  for fine tuning
mobileNet-adam-mixup-pruned-60-70 (half) : score = 0.2266, acc = 89.460%
mobileNet-adam-mixup-pruned-60-70 : score = 0.4531, acc = 89.480%


Batch : 64  for fine tuning
mobileNet-adam-mixup-pruned-60-65 (half) : score = 0.2366, acc = 90.320%
mobileNet-adam-mixup-pruned-60-65 : score = 0.4732, acc = 90.420%

Batch : 64 
lightNet-adam : score : Score = 0.0968, acc = 91.240%
lightNet-adam (half) : score = 0.0484, acc = 91.240%
lightNet-adam (8bits) : score =: 0.0242, acc = 90.730%
lightNet-adam (6bits) : score = 0.0181, acc = 91.420%

lightNetDepth : score = 0.0283, acc = 90.400 %, p : 0.0068, ops : 0.0214
lightNetDepth (6 bits) : score : 0.0053, acc = 90.440, p : 0.0013, ops : 0.0040