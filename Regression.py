import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import least_squares


class LinearRegression():
    """
    Linear Regression with constraint bounds
    It offered MSE and MAE as loss function, and both L1 and L2 regularization

    Parameters
    ----------
    w0:array, shape(n_features, )
        Initial value of weight vector

    bounds:2D tuple, (array,array) shape(n_features, ), default (-inf,inf)
        (lower bounds array, upper bounds array) each element in array indicate bound of weight
        for example, for a 2D weight vector [w0,w1], bounds=([-1,-2],[1,2]), w0~(-1,1) w1~(-2,2)

    loss:str, 'MSE', 'MAE', default 'MSE'
        2 kinds of loss function
            MSE=sum((Y-y)^2)
            MAE=sum(|Y-y|)

    regu:str, 'L1', 'L2' or None, default None
        2 kinds of Regularization
            L1=sum(|W|)
            L2=sum(W^2)

    C:float, default 0.01
        Coefficient of regularization

    Attributes
    ----------
    coef_ : array, shape (n_features, )
        Estimated coefficients for the linear regression problem.

    active_mask:array, shape (n_features,)
        Each component shows whether a corresponding constraint is active (that is, whether a variable is at the bound):
            0 : a constraint is not active.
            -1 : a lower bound is active.
            1 : an upper bound is active.

    cost:float
        Value of the cost function at the solution.

    fun:array, shape (n_samples,)
        Vector of residuals at the solution.


    Examples
    --------
    >>> import numpy as np
    >>> from Regression import LinearRegression
    >>> X=np.array([[1,2],[2,4]])
    >>> y=np.array([4,8])
    >>> w0=np.array([1,1])
    >>> Lr=LinearRegression()
    >>> Lr=LinearRegression(w0)
    >>> Lr.fit(X,y)
    <Regression.LinearRegression at 0x1082ae668>
    >>> Lr.predict(X)
    array([ 4.00000002,  8.00000003])
    >>> Lr.coef_
    array([ 1.2       ,  1.40000001])
    >>> lb=[-np.inf,-np.inf]
    >>> up=[1,1]
    >>> Lr=LinearRegression(w0,bounds=(lb,up))
    >>> Lr.fit(X,y)
    <Regression.LinearRegression at 0x108410ef0>
    >>> Lr.predict(X)
    array([ 3.,  6.])
    >>> Lr.coef_
    array([ 1.,  1.])

    Notes
    -----
    From the implementation point of view, this is based on Scipy Least Squares
    (scipy.optimize.least_squares). Wrapped it with some usual loss functions
    and regularization as a predictor object. And the coding style and method
    is refer to Sklearn
    """

    def __init__(self,w0,bounds=(-np.inf,np.inf),loss='MSE',regu=None,C=0.01,):

        self.w0 = w0
        self.bounds=bounds
        self.loss=loss
        self.regu = regu
        self.C=C

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array_like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """

        def func(w):

            MSE=np.dot(X,w)-y
            MAE=np.abs(np.dot(X, w) - y)**0.5

            L1=np.sum(np.abs(w))/X.shape[0]
            L2=np.sum(w**2)/X.shape[0]

            if self.loss == 'MSE':
                l=MSE

            if self.loss == 'MAE':
                l=MAE

            if self.regu=='L1':
                l=l+self.C*L1

            if self.regu=='L2':
                l=l+self.C*L2

            return l


        res = least_squares(func, self.w0, bounds=self.bounds)

        self.active_mask=res.active_mask
        self.cost=res.cost
        self.fun=res.fun
        self.coef_=res.x

        return self

    def predict(self,X):

        """
        Predict regression value of samples

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_: Predict value of samples
        """

        y_=np.dot(X,self.coef_)

        return y_

