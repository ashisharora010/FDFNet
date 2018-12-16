# FDFNet

This README will guide you through running the code and getting classification results on test images. The paper which discuss its detailed implementation is "FDFNet : A Secure Cancelable Deep Finger Dorsal Template Generation Network Secured via. Bio-Hashing" accepted at IEEE 5th International Conference on Identity, Security, and Behavior Analysis (ISBA) 2019.

## Steps ##
* Create a directory 'data' and put all the test images in their respective folders corresponding to their differnet classes.
* Run 'test-major.py'. This will create templates for all the test images and will save it to a .txt file.
* Run 'create_token.py'. This will create unique tokens for each subject.
* Run 'hascode_major.py'. Classification Accuracy will be shown at the end for all the images.
- - - -
Note:: train_templates and weights have been provided.

