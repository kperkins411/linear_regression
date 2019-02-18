# linear and Logistic Regression demos
Simple linear and logistic regression demos.  Automatically calculates maximum converging learning rate. Plots calculated learning rate.  Dynamically plots regression.<br>

Maximum converging learning rate calculated using techniques annotated in:
<a href="https://arxiv.org/abs/1506.01186"> 'Cyclical Learning Rates for Training Neural Networks' by Leslie Smith</a><br>
The function find_learning_rate(...)automatically calculates, plots, and returns the best learning rate.  Here is a sample plot
![My image](https://github.com/kperkins411/linear_regression/blob/master/art/lr_finder.png)

Once the learning rate is determined.  The function main(...) runs a linear regressor on a synthetic dataset and dynamicly plots the ongoing results.  A sample screenshot<br>
![My image](https://github.com/kperkins411/linear_regression/blob/master/art/regression.png)
