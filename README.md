# FDFNet : A Secure Cancelable Deep Finger Dorsal Template Generation Network Secured via. Bio-Hashing

## Introduction: ##  
This project corresponds to the work that has been accepted at IEEE 5th International Conference on Identity, Security, and Behavior Analysis (ISBA) 2019. In this work, we have proposed a secure cancelable finger dorsal template generation network (learning domain specific features) secured via. Bio-Hashing. Proposed system effectively protects the original finger dorsal images by withdrawing compromised template and reassigning the new one. A novel FingerDorsal Feature Extraction Net (FDFNet) has been proposed for extracting the discriminative features. This network is exclusively trained on trait specific features without using any kind of pre-trained architecture. Later Bio-Hashing, a technique based on assigning a tokenized random number to each user, has been used to hash the features extracted from FDFNet. To test the performance of the proposed architecture, we have tested it over two benchmark public finger knuckle datasets: PolyU FKP and PolyU Contactless FKI. The experimental results shows the effectiveness of the proposed system in terms of security and accuracy.

Finger dorsal feature extraction network (FDFNet) to learn domain specific features is depicted in the following figure:

![alt text](https://github.com/ashisharora010/FDFNet/blob/master/images/image2.png "Logo Title Text 1") 

Multishot enrollment architecture for cancelable finger dorsal biometric system is depicted in the following figure.  It is broadly divided into three subparts. The first part (Aggregated deep feature module) mainly deals with extracting finger dorsal specific features. The second subpart user token generation module is used to assign a unique identifier to each subject while the third part deals with generating cancelable template and its storage. 

![alt text](https://github.com/ashisharora010/FDFNet/blob/master/images/image1.png "Logo Title Text 2")
For more details kindly refer our paper  https://arxiv.org/pdf/1812.05308.pdf
Here, in this project we are providing our trained model on PolyU Contactless FKI Major Knucke Dataset which contains 503 subjects. Out of 5 images for each subject, 3 are considered for training and 2 for testing. Templates generated from training samples has been attached here. We are also providing some sample data corresponding to 3 subjects only.

## Steps to run the code ##
* Create a directory 'data' and put all the test images in their respective folders corresponding to their classes.
* Run 'test-major.py'. This will create templates for all the test images and will save it to a .txt file.
* Run 'create_token.py'. This will create unique tokens for each subject.
* Run 'hascode_major.py'. Classification Accuracy will be shown at the end for all the images.
- - - -

## Dependencies ##
* Python
* Numpy(1.14.5)
* PIL(1.1.7)
* scikit-learn(0.18.1)
* TensorFlow(1.8.0)
- - - -


