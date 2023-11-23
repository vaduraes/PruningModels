Fast Start:

To test models go to EvaluateModels.ipynb

#######################   On Tools:

###Load Datasets:
CIFAR10_LOADTER.py: Load CIFAR10 Dataset
TINYIMAGENET_LOADER.py: Load TINYIMAGENET Dataset

###Model Architectures:
ResNet18_CIFAR10.py: ResNet18 Architecture Adjusted for CIFAR10 images
VGG16_CIFAR10.py: VGG16 Architecture Adjusted for CIFAR10 images
ResNet18_TINYIMAGENET.py: ResNet18 Architecture Adjusted for TINYIMAGENET images
VGG16_TINYIMAGENET.py: VGG16 Architecture Adjusted for TINYIMAGENET images

###Evaluate Models
ErrorMetrics.py: Compute TOP 1 Error

#------------------------------------------------------------------------------------------


######################## .ipynb files
EvaluateModels.ipynb: Evaluate Trained models

PreProcessTinyImagenet.ipynb: Modify tmagenet dataset to a format easier to read by pytorch 

ResNet18_CIFAR.ipynb: Train a ResNet18 on CIFAR10 from scratch 
ResNet18_TINYIMAGENET.ipynb: Train a ResNet18 on TinyImagenet from scratch 

VGG16_CIFAR10.ipynb: Train a ResNet18 on CIFAR10 from scratch 
VGG16_TINYIMAGENET.ipynb: Train a ResNet18 on TinyImagenet from scratch 

#------------------------------------------------------------------------------------------
######################## On Networks:
###Stable_UNPRUNED
Stable model weights for ResNet18 and VGG16 on CIFAR10 and TinyImagenet (.pt)
Saved train/test loss progess (.npz)


######################## On Datasets:


Ideas:
Train with a cost function that penalizes class imballance
Train with OOD data as well