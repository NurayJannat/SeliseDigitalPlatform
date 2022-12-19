## Problem 2 
Two pretrained model has been used: VGG16 and VGG19.

### Inference:
#### VGG16: 
-  first 5 epochs with **steps_per_epoch**=20
-  then, 15 epochs with  **steps_per_epoch**=50
-  At last step, model was with
	- loss: 0.9368 
	-  acc: 0.6100 
	- val_loss: 0.7676 
	- val_acc: 0.7738

#### VGG19: 
-  10 epochs with **steps_per_epoch**=20
-  At last step, model was with
	- loss: 1.1533
	-  acc:  0.4913
	- val_loss: 0.9971 
	- val_acc: 0.6831

> Note: If more epochs are run, accuracy will be increasing. 

#### Ensemble technique
A basic ensemble technique was used: **Weighted Average Ensemble Technique**
As, second model (VGG19) was not trained with large epochs, so weight for first model (VGG16) was 0.60 or 60% and weight for second model (VGG19) was 0.40 or 40%.
Then, average of weighted result was taken and selected the class id with highest value.

### Training
Only VGG16 model has been provided to train with any dataset (following the structure).
By default, epochs =10 and steps_per_epoch=100 have been set.

