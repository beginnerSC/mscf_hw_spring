90 Second Presentation 
----

As Jonathan was saying, VIX is usually used to estimate the future 1 month realized variance of the index. Precisely the estimator is the formula up here. This is given by CBOE's definition of VIX. 

Here :math:`O` is the out of the money 1 month option prices. 

The only thing in this formula that's changing all the time is the option prices and the strikes, which I marked in the red color. Carr et al. use these sort of transformed option prices as the features. 

So we have features. We also need the labels for training. 

As mentioned earlier, there are regression I and regression II in the paper. They have the same features, only the labels are different. For regression I, the label is just the realized variance of the index. 

For regression II, the label is this VRP defined down here, which is a VIX-like index squared minus the realized variance. They defined this so called synthetic VIX, which is this VIX* in the formula. 

The synthetic VIX is very similar to COBE's VIX, the only difference is the number of options included in the summation formula up here. 



----


Our data is all from Bloomberg. 

We can only get implied volatility at specific delta, which we then plug into the Black-Scholes formula to get the options prices as the features. 

We had to back out the strikes from the delta, using the Black-Scholes delta formula. 

Next we will hand over to Raj to talk about the models. 
