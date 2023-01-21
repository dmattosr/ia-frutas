Log del Entrenamiento
=====

```

Number of Samples in Train:  84
Number of Samples in Valid:  9
Number of Samples in Test:  5
Total:  98
Number of Classes:  2
['buenas', 'malas']
cpu
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Conv2d: 1-1                            9,408
├─BatchNorm2d: 1-2                       128
├─ReLU: 1-3                              --
├─MaxPool2d: 1-4                         --
├─Sequential: 1-5                        --
|    └─BasicBlock: 2-1                   --
|    |    └─Conv2d: 3-1                  36,864
|    |    └─BatchNorm2d: 3-2             128
|    |    └─ReLU: 3-3                    --
|    |    └─Conv2d: 3-4                  36,864
|    |    └─BatchNorm2d: 3-5             128
|    └─BasicBlock: 2-2                   --
|    |    └─Conv2d: 3-6                  36,864
|    |    └─BatchNorm2d: 3-7             128
|    |    └─ReLU: 3-8                    --
|    |    └─Conv2d: 3-9                  36,864
|    |    └─BatchNorm2d: 3-10            128
|    └─BasicBlock: 2-3                   --
|    |    └─Conv2d: 3-11                 36,864
|    |    └─BatchNorm2d: 3-12            128
|    |    └─ReLU: 3-13                   --
|    |    └─Conv2d: 3-14                 36,864
|    |    └─BatchNorm2d: 3-15            128
├─Sequential: 1-6                        --
|    └─BasicBlock: 2-4                   --
|    |    └─Conv2d: 3-16                 73,728
|    |    └─BatchNorm2d: 3-17            256
|    |    └─ReLU: 3-18                   --
|    |    └─Conv2d: 3-19                 147,456
|    |    └─BatchNorm2d: 3-20            256
|    |    └─Sequential: 3-21             8,448
|    └─BasicBlock: 2-5                   --
|    |    └─Conv2d: 3-22                 147,456
|    |    └─BatchNorm2d: 3-23            256
|    |    └─ReLU: 3-24                   --
|    |    └─Conv2d: 3-25                 147,456
|    |    └─BatchNorm2d: 3-26            256
|    └─BasicBlock: 2-6                   --
|    |    └─Conv2d: 3-27                 147,456
|    |    └─BatchNorm2d: 3-28            256
|    |    └─ReLU: 3-29                   --
|    |    └─Conv2d: 3-30                 147,456
|    |    └─BatchNorm2d: 3-31            256
|    └─BasicBlock: 2-7                   --
|    |    └─Conv2d: 3-32                 147,456
|    |    └─BatchNorm2d: 3-33            256
|    |    └─ReLU: 3-34                   --
|    |    └─Conv2d: 3-35                 147,456
|    |    └─BatchNorm2d: 3-36            256
├─Sequential: 1-7                        --
|    └─BasicBlock: 2-8                   --
|    |    └─Conv2d: 3-37                 294,912
|    |    └─BatchNorm2d: 3-38            512
|    |    └─ReLU: 3-39                   --
|    |    └─Conv2d: 3-40                 589,824
|    |    └─BatchNorm2d: 3-41            512
|    |    └─Sequential: 3-42             33,280
|    └─BasicBlock: 2-9                   --
|    |    └─Conv2d: 3-43                 589,824
|    |    └─BatchNorm2d: 3-44            512
|    |    └─ReLU: 3-45                   --
|    |    └─Conv2d: 3-46                 589,824
|    |    └─BatchNorm2d: 3-47            512
|    └─BasicBlock: 2-10                  --
|    |    └─Conv2d: 3-48                 589,824
|    |    └─BatchNorm2d: 3-49            512
|    |    └─ReLU: 3-50                   --
|    |    └─Conv2d: 3-51                 589,824
|    |    └─BatchNorm2d: 3-52            512
|    └─BasicBlock: 2-11                  --
|    |    └─Conv2d: 3-53                 589,824
|    |    └─BatchNorm2d: 3-54            512
|    |    └─ReLU: 3-55                   --
|    |    └─Conv2d: 3-56                 589,824
|    |    └─BatchNorm2d: 3-57            512
|    └─BasicBlock: 2-12                  --
|    |    └─Conv2d: 3-58                 589,824
|    |    └─BatchNorm2d: 3-59            512
|    |    └─ReLU: 3-60                   --
|    |    └─Conv2d: 3-61                 589,824
|    |    └─BatchNorm2d: 3-62            512
|    └─BasicBlock: 2-13                  --
|    |    └─Conv2d: 3-63                 589,824
|    |    └─BatchNorm2d: 3-64            512
|    |    └─ReLU: 3-65                   --
|    |    └─Conv2d: 3-66                 589,824
|    |    └─BatchNorm2d: 3-67            512
├─Sequential: 1-8                        --
|    └─BasicBlock: 2-14                  --
|    |    └─Conv2d: 3-68                 1,179,648
|    |    └─BatchNorm2d: 3-69            1,024
|    |    └─ReLU: 3-70                   --
|    |    └─Conv2d: 3-71                 2,359,296
|    |    └─BatchNorm2d: 3-72            1,024
|    |    └─Sequential: 3-73             132,096
|    └─BasicBlock: 2-15                  --
|    |    └─Conv2d: 3-74                 2,359,296
|    |    └─BatchNorm2d: 3-75            1,024
|    |    └─ReLU: 3-76                   --
|    |    └─Conv2d: 3-77                 2,359,296
|    |    └─BatchNorm2d: 3-78            1,024
|    └─BasicBlock: 2-16                  --
|    |    └─Conv2d: 3-79                 2,359,296
|    |    └─BatchNorm2d: 3-80            1,024
|    |    └─ReLU: 3-81                   --
|    |    └─Conv2d: 3-82                 2,359,296
|    |    └─BatchNorm2d: 3-83            1,024
├─AdaptiveAvgPool2d: 1-9                 --
├─Linear: 1-10                           2,565
=================================================================
Total params: 21,287,237
Trainable params: 21,287,237
Non-trainable params: 0
=================================================================
Starting Epoch 1
Starting Epoch 2
Starting Epoch 3
Starting Epoch 4
Starting Epoch 5
No. epochs: 5 	Training Loss: 0.019 	Valid Loss 0.054 	Valid Accuracy 1.0
Starting Epoch 6
Starting Epoch 7
Starting Epoch 8
Starting Epoch 9
Starting Epoch 10
No. epochs: 10 	Training Loss: 0.001 	Valid Loss 0.003 	Valid Accuracy 1.0
Starting Epoch 11
Starting Epoch 12
Starting Epoch 13
Starting Epoch 14
Starting Epoch 15
No. epochs: 15 	Training Loss: 0.008 	Valid Loss 0.458 	Valid Accuracy 0.889
Starting Epoch 16
Starting Epoch 17
Starting Epoch 18
Starting Epoch 19
Starting Epoch 20
No. epochs: 20 	Training Loss: 0.002 	Valid Loss 0.0 	Valid Accuracy 1.0
Starting Epoch 21
Starting Epoch 22
Starting Epoch 23
Starting Epoch 24
Starting Epoch 25
No. epochs: 25 	Training Loss: 0.007 	Valid Loss 0.001 	Valid Accuracy 1.0
Starting Epoch 26
Starting Epoch 27
Starting Epoch 28
Starting Epoch 29
Starting Epoch 30
No. epochs: 30 	Training Loss: 0.002 	Valid Loss 0.0 	Valid Accuracy 1.0
Starting Epoch 31
Starting Epoch 32
Starting Epoch 33
Starting Epoch 34
Starting Epoch 35
No. epochs: 35 	Training Loss: 0.0 	Valid Loss 0.0 	Valid Accuracy 1.0
Starting Epoch 36
Starting Epoch 37
Starting Epoch 38
Starting Epoch 39
Starting Epoch 40
No. epochs: 40 	Training Loss: 0.0 	Valid Loss 0.001 	Valid Accuracy 1.0
Starting Epoch 41
Starting Epoch 42
Starting Epoch 43
Starting Epoch 44
Starting Epoch 45
No. epochs: 45 	Training Loss: 0.0 	Valid Loss 0.045 	Valid Accuracy 1.0
Starting Epoch 46
Starting Epoch 47
Starting Epoch 48
Starting Epoch 49
Starting Epoch 50
No. epochs: 50 	Training Loss: 0.0 	Valid Loss 0.053 	Valid Accuracy 1.0

```