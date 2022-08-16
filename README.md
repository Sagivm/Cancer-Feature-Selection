# Cancer-Feature-Selection

Here i will explain how to use the code and how to activate its learning and feature selection algorithm

I wrote the code for all FS methods and learning algortihms in a single standard so if one would understand how to use one he would understand all of them 

***NOTE***
All of the code should be run from the base folder of the project otherwise the paths for the dataset files wont fit\

## FS methods
The code consists of 7 feature selection methods listed in the feature selction directory. All of the code for a specific feature selection method is written in the corresponding file except for the GA_SVM, COMSVM-FRFE and COMESVM-FRFE methods that have other code that exist in the util directory

Running the code can be explained by this example:
![Alt text](/img/fs_method.png "Optional Title")

we simply need to insert to the algorithm tve full X aand y data with how many k features we want to keep


## Learnign method
The code consists of 5 learning algortihms listed in the classification directory. All file are written the same with the only difference is the calssification method being chosen
Here i will exaplin how to run and alter the algorithm values
we will use svm.py as an example algorithm

#### algorithm parameters
![Alt text](/img/params_ml.png "Optional Title")

The algorithm's parameters correspond to the algorithm of the FS method with an addition of the use of the augmentation of PCA and SMOTE

### Choosing a dataset
In order to choose a adtaset and activate the algorithm on is simple look at the main function
![Alt text](/img/main_ml.png "Optional Title")

When you want to change a dataset just change the read function to the one you want, all of them already imported

#### Choosing the FS method
![Alt text](/img/FS_ml.png "Optional Title")
Choosing a FS method is done by calling the FS method we listed before, in the for loop we have a list of all the FS methods we want to activarte on the data
for example if we want to activate mRMR and the RFE we would write
```python
 for selection_func in [("mrmr", mrmr_fs),("rfe", rfe_fs)]
```
#### Choosing the CV method
![Alt text](/img/CV_ml.png "Optional Title")
Choosing a CV method is done by choosing between loo and Kfold CV. The detemining CV method is the one listed in the second for method 

