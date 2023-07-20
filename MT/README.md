# Multi Tasking with Task-wise Split Gradient Boosting Trees

This is is the implementation of the _Task-wise split gradient boosting trees for multi-center diabetes prediction. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (pp. 2663-2673)._ Their implementation code for this is found [Here](https://github.com/felixwzh/MT-GBDT "GitHub MT-GBDT").

## Installation

To use this you will have to create an isolated conda enviornment usnig the pre-reated enviornment.yml.

```
conda env create -f environment.yml
conda activate mtenv
```

Then you need to set the enviornment up with the make file.

```
make	
cd python-package	
python setup.py install

```
You need to make sure that you have a compatible version of python(2.7), gcc, and GLIBCXX (_3.4.30) before setting the enviornment up.




