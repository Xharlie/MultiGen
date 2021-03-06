# MultiGen (Boosting visual concept generation by incorporating Generative Adversarial Network and Generative Cooperative Network)<br /> 
Paper: Generative Cooperative Net for Image Generation and Data Augmentation <br /> 
Authors: Qiangeng Xu, Zengchang Qin, Tao Wan <https://arxiv.org/abs/1705.02887> <br /> 
## 1. Introduction:
This is the development repository of the research 'MultiGen'.  
The goal of this research is twofold:
> A. To create a deconvolutional generative network that could learn the features' concept by training with sufficient numbers of picture-features pairs. 
> for example, using the MNIST dataset, ideally with the input of feature arrays(digit, handwritting style, color, transformation) and corresponding images of digits, we could not only train a deconv net to generate an identical image given the features that included in the training set, but also, generate the correct image out of the training set.   
 
> B. By using the mechanism of GANs, we would make an adversarial discriminator to play a min max game to make the result closer to the real-world photo, meanwhile, we should also be able to add a cooperative discriminator that undertake the multi-task-learning objective to classify the generated image to its feature classes. We would expect the discriminators to further improve the quality of the image generation. Besides, we would compare the discriminator with the classifier-alone MTL network. Since the discriminator might be able to outperform the solely classification network under the situation of data-starvation.  
For The Concept of Generative Cooperative Network: Here we use the multi-task Alex Net as the cooperative discriminator
</br>![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GCN.png)
  
  
## 2. Progress: 
### (2.1) Face
We have already successfully finished the step A by implementing the following structure in Caffe:
![Image of Deconv](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/Deconv.png) 
 
The features we have in our training set are: MMI
> 1. person: actor 0 to 6, using one-hot encoding: (0,0,0,1,0,0,0) person 3
> 2. emotion: each actor we choose 7 different emotion  one-hot encoding: (0,0,0,0,0,1,0) represent emotion 5, cheer
> 3. transformation: we have no tranformation, rotate 90 degree, rotate 180, rotate 270, mirror against X axis, mirror against Y axis.
(0,0,0,0,1,0) means mirror against X axis   
</br> 
ANOTHER larger Dataset we used is KDEF, which contain 70 actors with 7 facial expression.
 
## Our result's resolution is 158*158, which is larger and more difficult than almost all other state-of-the-art generative network!! 
  
  
Result 1:  
> Images that are contained in the training set:
</br>Ground Truth:</br>
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-0.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-1.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-2.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-3.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-4.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-5.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT0-6.png)
</br>Generated:</br>
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-0.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-1.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-2.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-3.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-4.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-5.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN0-6.png)

Result 2:  
> Images aren't included but images with same person, emotion and transformation are included in the training set:  
This experiment is testing the network's capability to learn the manifold of a person and one of the particular transformation.
Since MMI's dataset is too small and make it hard to get a clear rotation generation result, here, we use KDEF's result to demonstrate.
</br>Ground Truth:</br>
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt4-0-1.JPG)
</br>Generated:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn4-0-1.JPG)    



Result 3:  
> Images aren't included in the training set, but same actor with different emotions and other actors with same emotion are included in the training set:  
We can see how our network is able to learn the manifold of an individual and a specific emotion from other individuals. 
Meanwhile, we can see the generated result is a bit fuzzy and we are working toward a result that closer to the real-world photo. 
Besides, we believe implement an GANs on phase two can further enable us on this endeavour.

<br>Very small dataset: MMI </br>
</br>Ground Truth:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT3-3-0.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT3-3-1.png) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT3-3-2.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GT3-3-3.png)
</br>Generated:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN3-3-0.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN3-3-1.png) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN3-3-2.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/face/GEN3-3-3.png)  
</br> We can see that the result is fuzzy, but the expression concept of supprise is preliminarily captured.
<br> larger dataset: KDEF </br>
<br>(1)Concept generation: Supprise
</br>Ground Truth:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt37-2-0.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt37-2-1.JPG) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt37-2-2.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt37-2-3.JPG)
</br>Generated:(we can clearly see since average actor's mouth doesn't open as large as this actor, the generated supprising expression for this actor has average opened mouth and personal facial feature preserved.)</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn37-2-0.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn37-2-1.JPG) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn37-2-2.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn37-2-3.JPG)
<br><br>(2)Concept generation: Happy
</br>Ground Truth:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt56-1-0.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt56-1-1.JPG) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt56-1-2.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt56-1-3.JPG)
</br>Generated: happy face by far has the best performance among other expressions.</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn56-1-0.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn56-1-1.JPG) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn56-1-2.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn56-1-3.JPG)
<br><br>(3)Concept generation: Aversion
</br>Ground Truth:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt55-0-0.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt55-0-1.JPG) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt55-0-2.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gt55-0-3.JPG)
</br>Generated: since different actors' aversion expression always have different grin direction, the generated grin looks a little surreal for this person</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn55-0-0.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn55-0-1.JPG) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn55-0-2.JPG)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/KDEF/gn55-0-3.JPG)
</br>
### (2.2) Digit
We have already successfully finished the step A by implementing the following structure in Caffe:
![Image of Deconv](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/Deconv.png)

The features we have in our training set are:
> 1. digit: digit 0 to 9, using one-hot encoding: (0,0,0,1,0,0,0,0,0,0) represent 3
> 2. style: each digit we choose 10 different writing styles, using  one-hot encoding: (0,0,0,0,0,1,0,0,0,0) represent style 5
> 3. color: we have red, green, blue three colors for each picture, (0,1,0) represent green
> 4. transformation: we have no tranformation, rotate 90 degree, rotate 180, rotate 270, mirror against X axis, mirror against Y axis.
(0,0,0,0,1,0) means mirror against X axis  
  
Result 1:  
> Images that are contained in the training set:
</br>Ground Truth:</br> 
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT3210.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT3211.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT3212.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT3213.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT3214.png)
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT3215.png)
</br>Generated:</br> 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN3210.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN3211.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN3212.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN3213.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN3214.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN3215.png)  

Result 2:  
> Images aren't contained but images with same digit, style and transformation are included in the training set:
</br>Ground Truth:</br>
![Image of GT](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT2401.png)
</br>Generated:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN2401.png)    
  
Result 3:  
> Images aren't contained but images with same digit and style are included in the training set:  
Since the specific style we chosen on this 7 is not consist through all other digits, the network cannot learn the  
writting style but only be able to learn a average 7 as a concept of digit.
</br>Ground Truth:</br>
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT7904.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT7914.png) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GT7924.png)
</br> Generated:</br> 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN7904.png)
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN7914.png) 
![Image of GEN](https://github.com/Xharlie/MultiGen/blob/master/README_IMG/digit/GEN7924.png) 
  
  
# NextStep:     
> 1. Prepare an emotion, people multi label classification discriminator as a cooperative partner. Add the classification 
loss as part of generation loss.
  
> 2. Add an adversarial discriminator network at the end of our generative network. 
By doing these, we can explore the ability of GANs to provide better generative result. 

> 3. Evaluate the GCN's classication accuracy and compared it with a regular classification network that have same structure.

