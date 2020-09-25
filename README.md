# DropBlock

![build](https://travis-ci.org/miguelvr/dropblock.png?branch=master)


Implementation of [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf) 
in PyTorch.

## Abstract

Deep neural networks often work well when they are over-parameterized 
and trained with a massive amount of noise and regularization, such as 
weight decay and dropout. Although dropout is widely used as a regularization 
technique for fully connected layers, it is often less effective for convolutional layers. 
This lack of success of dropout for convolutional layers is perhaps due to the fact 
that activation units in convolutional layers are spatially correlated so 
information can still flow through convolutional networks despite dropout. 
Thus a structured form of dropout is needed to regularize convolutional networks. 
In this paper, we introduce DropBlock, a form of structured dropout, where units in a 
contiguous region of a feature map are dropped together. 
We found that applying DropBlock in skip connections in addition to the 
convolution layers increases the accuracy. Also, gradually increasing number 
of dropped units during training leads to better accuracy and more robust to hyperparameter choices. 
Extensive experiments show that DropBlock works better than dropout in regularizing 
convolutional networks. On ImageNet classification, ResNet-50 architecture with 
DropBlock achieves 78.13% accuracy, which is more than 1.6% improvement on the baseline. 
On COCO detection, DropBlock improves Average Precision of RetinaNet from 36.8% to 38.4%.

## Usage
For 2D input only
```python
from dropblock import DropBlock

class ResNet(nn.Module):
    def __init__(self, block, n_blocks):
        super().__init__()
        ... 
        # somewhere in init method
        self.layer1 = self._make_layer(block, n_blocks[0], 64, 2)
        self.layer2 = self._make_layer(block, n_blocks[1], 160, 2)
        self.layer3 = self._make_layer(block, n_blocks[2], 320, 2)
        self.layer4 = self._make_layer(block, n_blocks[3], 640, 2)

        ### INIT DropBlock layers
        self.dropblock = DropBlock(drop_prob=0.1, block_size=5, warmup_steps=2500)
        ###
        ...

    def forward(self, x):
        ...
        # somewhere in forward method
        x = self.layer1(x)
        x = self.layer2(x)

        ### CALL DropBlock
        x = self.dropblock(self.layer3(x))
        x = self.dropblock(self.layer4(x))
        ###
        ...
```     
```python
# Finally in your training loop
def training_step(self, batch):
    img, label = batch
    
    ### STEP the warm up period
    self.model.dropblock.step()
    ###

    y = self.model(img)
    loss = self.xcent_loss(z, label)
```