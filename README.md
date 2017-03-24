# MultiGen
## 1. Introduction:
This is the development repository of the research 'Multi-task Generative Network'.  
The goal of this research is twofold:
> A. To create a deconvolutional generative network that could learn the features' concept by training with sufficient numbers of picture-features pairs. 

> for example, using the MNIST dataset, ideally with the input of feature arrays(digit, handwritting form, color, transformation) and corresponding images of digits, we could not only train a deconv net to generate an identical image given the features that included in the training set, but also, generate the correct image out of the training set.  

> B. By using the mechanism of GANs, we should be able to add a discriminator that undertake the multi-task-learning objective to classify the generated image to its feature classes. We would expect the discriminator to further improve the quality of the image generation. Besides, we would compare the discriminator with the classifier-alone MTL network. Since the discriminator might be able to outperform the solely classification network under the situation of data-starvation.  
  
  
## 2. Progress:  
We have already successfully finished the step A by implementing the following structure in Caffe:
![Image of Deconv]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/Deconv.png)

The features we have in our training set are:
> 1. digit: digit 0 to 9, using one-hot encoding: (0,0,0,1,0,0,0,0,0,0) represent 3
> 2. form: each digit we choose 10 different form, using  one-hot encoding: (0,0,0,0,0,1,0,0,0,0) represent form 5
> 3. color: we have red, green, blue three colors for each picture, (0,1,0) represent green
> 4. transformation: we have no tranformation, rotate 90 degree, rotate 180, rotate 270, mirror against X axis, mirror against Y axis.
(0,0,0,0,1,0) means mirror against X axis  
  
Result 1:  
> images that are contained in the training set:  
GT:  
![Image of GT]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT3210.png)![Image of GT](https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT3211.png)![Image of GT]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT3212.png)![Image of GT]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT3213.png)![Image of GT]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT3214.png)![Image of GT]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT3215.png)  
Generated:  
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN3210.png)![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN3211.png)![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN3212.png)![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN3213.png)![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN3214.png)![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN3215.png)  

Result 2:  
> images aren't contained but images with same digit, form and transformation are included in the training set:  
GT:  
![Image of GT] 
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT2401.png)  
Generated:    
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN2401.png)    
  
Result 3:  
> images aren't contained but images with same digit and form are included in the training set:  
GT:  
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT7904.png)
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT7914.png) 
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GT7924.png)  
Generated:    
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN7904.png)
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN7914.png) 
![Image of GEN]
(https://github.com/Xharlie/MultiGen/blob/alterNet/README_IMG/GEN7924.png) 
  
  
# NextStep:     
> 1. Use MMI face expression dataset to further prove the deconv generate network's ability,  
to train on different individual's face expression and then hopefully we can generate expressions  
that is not included for one person (but of course existed for other individuals).
  
> 2. Add a discriminator network at the end of our generative network. The discriminator should also include  
classification loss. By doing these, we can explore the ability of MTL GANs to provide better generative result  
and even classification result(compared with standard classification network under data starvation).

