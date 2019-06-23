# predict-eps

Predict the EPS (earning per share) in th next year based on the features (e.g., liability, debt asset ratio, return on assets) in previous years. The data comes from https://github.com/1qweasdzxc/python/tree/master/%E4%BA%94%E5%A4%A7%E8%B4%A2%E5%8A%A1%E6%AF%94%E4%BE%8B, which extracts indicators from the financial statement of 3455 companies.

The input features ranges accross 4 years and we use RNN to train the model to memorize the historical information. The metric is the mean-square-error of the predicted EPS and the real EPS in the next year.

Then with the well-trained model, we can predict the EPS in the next year of one company, given the information of that company in previuos years.
