import numpy as np
import scipy.optimize as opt
import scipy.linalg as sla
import jax
jax.config.update("jax_enable_x64", True)
from   jax import numpy as jnp
import jax.scipy.linalg as jsla
from typing import NamedTuple
from scipy.optimize import OptimizeResult

class ParamOutput(NamedTuple):
    lam: float = np.nan
    alpha: float = np.nan
    d: float = np.nan
    opt_result: OptimizeResult = None

# =========================
# SECOND DIFFERENCE PENALTY
# =========================
def second_difference_matrix(p):
    D = np.zeros((p-2, p))
    for i in range(p-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    return D.T @ D

# =========================
# MSE
# =========================
def mse(y, yhat):
    return np.mean((y - yhat)**2)

# =========================
# OLS
# =========================
def ols(X,y, criteria = 'GCV'):
    """ Ordinary least squares
    Input: X, y
        Design matrix X (n x p), response y (size n)
        Optional: optimal lambda parameters,
            criteria is either 'GCV' for the Generalized Cross Validation criteria,
            or 'CV' to use leave-one-out (LOO) Cross Validation

    Output: beta_optimal, gcv
        where gcv is the generalized cross validation score
        or the Leave-One-Out (LOO) cross-validation score
    """
    n, p = X.shape
    beta = sla.lstsq(X,y)[0]
    yhat = X @ beta
    P    = X @ sla.lstsq(X,np.eye(n))[0]
    if criteria.lower() == 'gcv':
        gcv_ols = mse(y,yhat)/( (1-np.trace(P)/n)**2 )
    elif criteria.lower() == 'cv':
        gcv_ols = np.mean(((y-yhat)/(1-np.diag(P)))**2 ) 
    else:
        raise ValueError('Criteria must be either "GCV" or "CV"')
    return beta, gcv_ols, ParamOutput()

# =========================
# RIDGE
# =========================
def ridge(X, y, lam_opt = None, lambda_bounds = (1e-6,1e6), opt_method = 'SLSQP', 
          criteria = 'GCV', log_change_of_variables=None, penalize_constant=True):
    """ Classic ridge regression: $beta = (XX^T + lambda I)^{-1} X^T y$ 
    Input: X, y, [lam_opt]
        Design matrix X (n x p), response y (size n)
        Optional: optimal lambda parameters. If not provided, then optimal values
            (based on the GCV or LOO CV criteria) are found using a gradient-based
            optimization solver
            In this case, lambda_bounds are bound constraints on the size of lambda
            and opt_method is a string controlling which solver to use from scipy.optimize.minimize
            criteria is either 'GCV' for the Generalized Cross Validation criteria,
            or 'CV' to use leave-one-out (LOO) Cross Validation

            log_change_of_variables determines whether the optimizer uses a log change-of-variables.
                If set to None, then it automatically determines whether to do the change-of-variables
                based on whether lambda_bounds[1]/lambda_bounds[0] >= 1e3
            
            penalize_constant (default: True) determines whether I is the identity matrix,
                or if penalize_constant=False, then it is the identity matrix except the top left entry
                is zero (so that if the top entry of beta represents the constant term, 
                then the constant term isn't penalized)

    Output: beta_optimal, gcv, lambda_optimal
        where gcv is the generalized cross validation score (or LOO CV).
    """
    n, p = X.shape
    G  = jnp.asarray(X.T @ X)
    Xy = jnp.asarray(X.T @ y)
    X  = jnp.asarray(X)
    if log_change_of_variables is None:
        if lambda_bounds[1]/lambda_bounds[0] >= 1e3:
            log_change_of_variables = True
        else:
            log_change_of_variables = False
    if log_change_of_variables and (lam_opt is None):
        log = jnp.log
        exp = jnp.exp
    else:
        log = lambda x : x
        exp = lambda x : x
    I = jnp.eye(p)        
    if not penalize_constant:
        I[0,0] = 0.
        
    def gcv(lam_vec):
        """ Generalized Cross Validation criterion """
        lam = exp(lam_vec[0])
        A = G + lam*I
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve( (c,lower), Xy )
        yhat = jnp.asarray(X) @ beta
        P      = X @ jsla.cho_solve( (c,lower), X.T )
        traceP = jnp.trace(P)
        return jnp.mean((jnp.asarray(y)-yhat)**2)/((1-traceP/n)**2)
    
    def cv(lam_vec):
        """ Leave-one-out Cross Validation criterion, specialized for linear predictors """
        lam = exp(lam_vec[0])
        A = G + lam*I
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve( (c,lower), Xy )
        yhat = jnp.asarray(X) @ beta
        P      = X @ jsla.cho_solve( (c,lower), X.T )
        return jnp.mean(((jnp.asarray(y)-yhat)/(1-jnp.diag(P)))**2 ) # only difference from GCV
    
    if criteria.lower() == 'gcv':
        crit = gcv
    elif criteria.lower() == 'cv':
        crit = cv
    else:
        raise ValueError('Criteria must be either "GCV" or "CV"')
    
    if lam_opt is None:
        # We look for the optimal vector
        grad      = jax.jit( jax.grad(crit) )
        lambda_bounds = tuple(log(jnp.array(lambda_bounds)))
        lambda0 = (lambda_bounds[0] + lambda_bounds[1])/2 # midpoint
        results   = opt.minimize(crit, np.array([lambda0]), jac=grad,
                          bounds=[lambda_bounds], method=opt_method)
        lam_opt   = exp(results.x[0])
        beta      = sla.solve(X.T@X + lam_opt*np.eye(p), X.T@y,assume_a='pos')
        gcv_ridge = results.fun # already computed
    else:
        if isinstance(lam_opt, ParamOutput):
            lam_opt = lam_opt[0]
        beta = sla.solve(X.T@X + lam_opt*np.eye(p), X.T@y,assume_a='pos')
        gcv_ridge = crit(lam_opt)
        results = None
    return beta, gcv_ridge, ParamOutput(lam=lam_opt, opt_result = results)


# =========================
# CLASSICAL LIU
# =========================
def classical_liu(X, y, params_optimal = None, lambda_bounds = (1e-6,1e6), 
                  opt_method = 'SLSQP', d_bounds = (0,1), 
                  criteria = 'GCV', log_change_of_variables=None, penalize_constant=True):
    """ Standard liu: $beta = (XX^T + lambda I)^{-1}(X^T y + d lambda beta_{OLS})$
    Input: X, y, [params_optimal]
        Design matrix X (n x p), response y (size n)
        Optional: optimal parameters for d and lambda. If not provided, then optimal values
            (based on the GCV criteria) are found using a gradient-based
            optimization solver
            In this case, lambda_bounds are bound constraints on the size of lambda
            (the value of d is always constrained to be in (0,1))
            and opt_method is a string controlling which solver to use from scipy.optimize.minimize
            criteria is either 'GCV' for the Generalized Cross Validation criteria,
            or 'CV' to use leave-one-out (LOO) Cross Validation

            log_change_of_variables determines whether the optimizer uses a log change-of-variables.
                If set to None, then it automatically determines whether to do the change-of-variables
                based on whether lambda_bounds[1]/lambda_bounds[0] >= 1e3
            
            penalize_constant (default: True) determines whether I is the identity matrix,
                or if penalize_constant=False, then it is the identity matrix except the top left entry
                is zero (so that if the top entry of beta represents the constant term, 
                then the constant term isn't penalized)

    Output: beta_optimal, gcv, lambda_optimal, d_optimal
        where gcv is the generalized cross validation score (or LOO CV).
    """
    n, p = X.shape
    G  = jnp.asarray(X.T @ X)
    Xy = jnp.asarray(X.T @ y)
    X  = jnp.asarray(X)
    beta_OLS = jnp.asarray(sla.lstsq(X,y)[0])
    P_OLS = jnp.asarray(sla.lstsq(X,np.eye(n))[0])
    if log_change_of_variables is None:
        if lambda_bounds[1]/lambda_bounds[0] >= 1e3:
            log_change_of_variables = True
        else:
            log_change_of_variables = False
    if log_change_of_variables and (params_optimal is None):
        log = jnp.log
        exp = jnp.exp
    else:
        log = lambda x : x
        exp = lambda x : x
    I = jnp.eye(p)        
    if not penalize_constant:
        I[0,0] = 0.
        
    def gcv(params):
        lam, d = params
        lam = exp(lam)
        A = G + lam*I
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy + d*lam*beta_OLS)
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T + d*lam*P_OLS )
        traceP = jnp.trace(P)
        return jnp.mean((jnp.asarray(y)-yhat)**2)/((1-traceP/n)**2)

    def cv(params):
        lam, d = params
        lam = exp(lam)
        A = G + lam*I
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy + d*lam*beta_OLS)
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T + d*lam*P_OLS )
        return jnp.mean(((jnp.asarray(y)-yhat)/(1-jnp.diag(P)))**2 ) # only difference from GCV
    
    if criteria.lower() == 'gcv':
        crit = gcv
    elif criteria.lower() == 'cv':
        crit = cv
    else:
        raise ValueError('Criteria must be either "GCV" or "CV"')
    
    if params_optimal is None:
        grad = jax.jit( jax.grad(crit) )
        lambda_bounds = tuple(log(jnp.array(lambda_bounds)))
        lambda0 = (lambda_bounds[0] + lambda_bounds[1])/2 # midpoint
        d0 = (d_bounds[0]+d_bounds[1])/2 # midpoint
        res = opt.minimize(crit, np.array([lambda0,d0]), jac=grad,
                          bounds=[lambda_bounds,d_bounds], method=opt_method)

        lam_opt, d_opt = res.x
        lam_opt = exp(lam_opt)
        gcv_liu = res.fun
    else:
        if isinstance(params_optimal, ParamOutput):
            lam_opt = params_optimal.lam
            d_opt   = params_optimal.d
        else:
            lam_opt, d_opt = params_optimal
        gcv_liu = crit(params_optimal)
        res = None

    beta = sla.solve(X.T@X + lam_opt*np.eye(p),
                     X.T@y + d_opt*lam_opt*np.array(beta_OLS),
                     assume_a = 'pos')

    return beta, gcv_liu, ParamOutput(lam=lam_opt, d=d_opt, opt_result = res)

# =========================
# GENERALIZED RIDGE ESTIMATOR
# =========================
def generalized_ridge(X, y, R, params_optimal=None, lambda_bounds = (1e-6,1e6), opt_method = 'SLSQP', 
           criteria = 'GCV', log_change_of_variables=None, penalize_constant=True):
    """ generalized ridge regression estimator
        $beta = (XX^T + Q)^{-1} X^T y$
        where Q = lambda(alpha I + (1-alpha) R)
    Input: X, y, R, [params_optimal]
        Design matrix X (n x p), response y (size n)
        Optional: optimal parameters for d and lambda. If not provided, then optimal values
            (based on the GCV criteria) are found using a gradient-based
            optimization solver
            In this case, lambda_bounds are bound constraints on the size of lambda
            (the value of alpha is always constrained to be in (0,1))
            and opt_method is a string controlling which solver to use from scipy.optimize.minimize
            criteria is either 'GCV' for the Generalized Cross Validation criteria,
            or 'CV' to use leave-one-out (LOO) Cross Validation

            log_change_of_variables determines whether the optimizer uses a log change-of-variables.
                If set to None, then it automatically determines whether to do the change-of-variables
                based on whether lambda_bounds[1]/lambda_bounds[0] >= 1e3
            
            penalize_constant (default: True) determines whether I is the identity matrix,
                or if penalize_constant=False, then it is the identity matrix except the top left entry
                is zero (so that if the top entry of beta represents the constant term, 
                then the constant term isn't penalized)

    Output: beta_optimal, gcv, lambda_optimal, alpha_optimal
        where gcv is the generalized cross validation score (or LOO CV)
        and lambda and alpha were determined to minimize gcv (or LOO CV).
    """
    n, p = X.shape
    G  = jnp.asarray(X.T @ X)
    Xy = jnp.asarray(X.T @ y)
    X  = jnp.asarray(X)
    Rj = jnp.asarray(R)
    I  = jnp.eye(p)
    if not penalize_constant:
        I[0,0] = 0.
    if log_change_of_variables is None:
        if lambda_bounds[1]/lambda_bounds[0] >= 1e3:
            log_change_of_variables = True
        else:
            log_change_of_variables = False
    if log_change_of_variables and (params_optimal is None):
        log = jnp.log
        exp = jnp.exp
    else:
        log = lambda x : x
        exp = lambda x : x
        
    def gcv(params):
        lam, alpha = params
        lam = exp(lam)
        Q = lam*(alpha*I + (1-alpha)*Rj)
        A = G + Q
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy)
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T )
        traceP = jnp.trace(P)
        return jnp.mean((jnp.asarray(y)-yhat)**2)/((1-traceP/n)**2)

    def cv(params):
        lam, alpha = params
        lam = exp(lam)
        Q = lam*(alpha*I + (1-alpha)*Rj)
        A = G + Q
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy)
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T )
        return jnp.mean(((jnp.asarray(y)-yhat)/(1-jnp.diag(P)))**2 ) # only difference from GCV
    
    if criteria.lower() == 'gcv':
        crit = gcv
    elif criteria.lower() == 'cv':
        crit = cv
    else:
        raise ValueError('Criteria must be either "GCV" or "CV"')
    
    if params_optimal is None:
        grad = jax.jit( jax.grad(crit) )
        lambda_bounds = tuple(log(jnp.array(lambda_bounds)))
        lambda0 = (lambda_bounds[0] + lambda_bounds[1])/2 # midpoint
        res = opt.minimize(crit, np.array([lambda0,0.5]), jac=grad,
                          bounds=[lambda_bounds,(0,1)], method=opt_method)

        lam_opt, alpha_opt = res.x
        lam_opt = exp(lam_opt)
        gcv_cardot = res.fun
    else:
        if isinstance(params_optimal, ParamOutput):
            lam_opt, alpha_opt = params_optimal[:2]
        else:
            lam_opt, alpha_opt = params_optimal
        gcv_cardot = crit(params_optimal)
        res = None

    Q = lam_opt*(alpha_opt*np.eye(p) + (1-alpha_opt)*R)
    beta = sla.solve(X.T@X + Q, X.T@y, assume_a = 'pos')

    return beta, gcv_cardot, ParamOutput(lam=lam_opt, alpha=alpha_opt, opt_result = res)


# =========================
# FUNCTIONAL LIU
# =========================
def functional_liu(X, y, R, params_optimal=None, lambda_bounds = (1e-6,1e6), d_bounds = (0,1), 
                   opt_method = 'SLSQP', criteria = 'GCV', gridsize = 5, 
                   log_change_of_variables=None, penalize_constant=True):
    """ Functional Liu estimator: combination of generalized ridge with Liu's biased estimator
        $(XX^T + Q)^{-1}(X^T y + d lambda Q beta_{OLS})$
        where Q = lambda(alpha I + (1-alpha) R) as in Cardot.
    Input: X, y, R, [params_optimal]
        Design matrix X (n x p), response y (size n)
        Optional: optimal parameters for lambda, d, alpha. If not provided, then optimal values
            (based on the GCV criteria) are found using a gradient-based
            optimization solver
            In this case, lambda_bounds are bound constraints on the size of lambda
            (the value of d and alpha are always constrained to be in (0,1))
            and opt_method is a string controlling which solver to use from scipy.optimize.minimize
            criteria is either 'GCV' for the Generalized Cross Validation criteria,
            or 'CV' to use leave-one-out (LOO) Cross Validation

            log_change_of_variables determines whether the optimizer uses a log change-of-variables.
                If set to None, then it automatically determines whether to do the change-of-variables
                based on whether lambda_bounds[1]/lambda_bounds[0] >= 1e3
            
            penalize_constant (default: True) determines whether I is the identity matrix,
                or if penalize_constant=False, then it is the identity matrix except the top left entry
                is zero (so that if the top entry of beta represents the constant term, 
                then the constant term isn't penalized)

    Output: beta_optimal, gcv, lambda_optimal, d_optimal, alpha_optimal
        where gcv is the generalized cross validation score (or LOO CV)
        and lambda, d and alpha were determined to minimize gcv (or LOO CV).
    """
    n, p = X.shape
    G  = jnp.asarray(X.T @ X)
    Xy = jnp.asarray(X.T @ y)
    X  = jnp.asarray(X)
    beta_OLS = jnp.asarray(sla.lstsq(X,y)[0])
    Rj = jnp.asarray(R)
    I  = jnp.eye(p)
    if not penalize_constant:
        I[0,0] = 0.
    P_OLS = jnp.asarray(sla.lstsq(X,np.eye(n))[0])
    if log_change_of_variables is None:
        if lambda_bounds[1]/lambda_bounds[0] >= 1e3:
            log_change_of_variables = True
        else:
            log_change_of_variables = False
    if log_change_of_variables and (params_optimal is None):
        log = jnp.log
        exp = jnp.exp
    else:
        log = lambda x : x
        exp = lambda x : x
        
    def gcv(params):
        lam, d, alpha = params
        lam = exp(lam)
        Q = lam*(alpha*I + (1-alpha)*Rj)
        A = G + Q
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy + d*(Q@beta_OLS) )
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T + d*Q@P_OLS )
        traceP = jnp.trace(P)
        return jnp.mean((jnp.asarray(y)-yhat)**2)/((1-traceP/n)**2)
    
    def cv(params):
        lam, d, alpha = params
        lam = exp(lam)
        Q = lam*(alpha*I + (1-alpha)*Rj)
        A = G + Q
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy + d*(Q@beta_OLS) )
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T + d*Q@P_OLS )
        return jnp.mean(((jnp.asarray(y)-yhat)/(1-jnp.diag(P)))**2 ) # only difference from GCV
    
    if criteria.lower() == 'gcv':
        crit = gcv
    elif criteria.lower() == 'cv':
        crit = cv
    else:
        raise ValueError('Criteria must be either "GCV" or "CV"')
    
    if params_optimal is None:
        grad = jax.jit( jax.grad(crit) )
        lambda_bounds = tuple(log(jnp.array(lambda_bounds)))
        # First, do a coarse grid search
        if gridsize > 1:
            crit = jax.jit(crit)
            best_crit = np.inf
            best_param = (0,0,0)
            for lam in np.linspace(*lambda_bounds,gridsize):
                for d in np.linspace(*d_bounds,gridsize):
                    for alpha in np.linspace(0,1,gridsize):
                        params = (lam, d, alpha)
                        crit_value = crit(params)
                        if crit_value < best_crit:
                            best_crit = crit_value
                            best_param = params
        else:
            lambda0 = (lambda_bounds[0] + lambda_bounds[1])/2 # midpoint
            d0 = (d_bounds[0]+d_bounds[1])/2 # midpoint
            best_param = np.array([lambda0,d0,0.5])

        res = opt.minimize(crit, best_param, jac=grad,
                          bounds=[lambda_bounds,d_bounds,(0,1)], method=opt_method)

        lam_opt, d_opt, alpha_opt = res.x
        lam_opt = exp(lam_opt)
        gcv_fliu = res.fun
    else:
        if isinstance(params_optimal, ParamOutput):
            lam_opt, alpha_opt, d_opt = params_optimal[:3]
        else:
            lam_opt, d_opt, alpha_opt = params_optimal
        gcv_fliu = crit(params_optimal)
        res = None

    Q = lam_opt*(alpha_opt*np.eye(p) + (1-alpha_opt)*R)
    beta = sla.solve(X.T@X + Q,
                     X.T@y + d_opt*(Q@np.linalg.lstsq(X,y,rcond=None)[0]),
                     assume_a = 'pos')

    return beta, gcv_fliu, ParamOutput(lam=lam_opt, d=d_opt, 
                                       alpha=alpha_opt, opt_result = res)
