# Traditional-algorithms related to machine learning
    Personal practice notes are only used for learning
## Project specification
### `1. Auxiliary tool file - Auxi.py`
    It is used to generate random sample data and visualize the model effect
### `2. Adaboost`
* Additive model
    * Each round of classification results are weighted and then added to the final classification results
    * The sample weight of the next round is updated according to the classification error rate of the current round
### `3. GBDT`
* Additive model
    * Optimization is done in function space
    * Direction of negative gradient
    * In this example, we accumulate the model residuals in each iteration
### `4. Percrptron`
* Function margin
    * Stochastic gradient descent
### `5. Decision tree`
* Entropy
    * Information gain
### `6. GMMvsKmean`
* EM
    * update distribution of sample data
    * update parameters of the model
