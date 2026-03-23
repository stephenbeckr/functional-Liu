import numpy as np
import scipy.optimize as opt
import scipy.linalg as sla
import jax
jax.config.update("jax_enable_x64", True)
from   jax import numpy as jnp
import jax.scipy.linalg as jsla

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
            or 'CV' to use leave-one-out (LOO) Cross Valiation

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
    return beta, gcv_ols

# =========================
# RIDGE
# =========================
def ridge(X, y, lam_opt = None, lambda_bounds = (1e-6,1e6), opt_method = 'SLSQP', 
          criteria = 'GCV', log_change_of_variables=None):
    """ Classic ridge regression: $beta = (XX^T + lambda I)^{-1} X^T y$ 
    Input: X, y, [lam_opt]
        Design matrix X (n x p), response y (size n)
        Optional: optimal lambda parameters. If not provided, then optimal values
            (based on the GCV or LOO CV criteria) are found using a gradient-based
            optimization solver
            In this case, lambda_bounds are bound constraints on the size of lambda
            and opt_method is a string controlling which solver to use from scipy.optimize.minimize
            criteria is either 'GCV' for the Generalized Cross Validation criteria,
            or 'CV' to use leave-one-out (LOO) Cross Valiation

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
        
    def gcv(lam_vec):
        """ Generalized Cross Validation criterion """
        lam = exp(lam_vec[0])
        A = G + lam*jnp.eye(p)
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve( (c,lower), Xy )
        yhat = jnp.asarray(X) @ beta
        P      = X @ jsla.cho_solve( (c,lower), X.T )
        traceP = jnp.trace(P)
        return jnp.mean((jnp.asarray(y)-yhat)**2)/((1-traceP/n)**2)
    
    def cv(lam_vec):
        """ Leave-one-out Cross Validation criterion, specialized for linear predictors """
        lam = exp(lam_vec[0])
        A = G + lam*jnp.eye(p)
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
        results   = opt.minimize(crit, log(np.array([1.0])), jac=grad,
                          bounds=[lambda_bounds], method=opt_method)
        lam_opt   = exp(results.x[0])
        beta      = sla.solve(X.T@X + lam_opt*np.eye(p), X.T@y,assume_a='pos')
        gcv_ridge = results.fun # already computed
    else:
        beta = sla.solve(X.T@X + lam_opt*np.eye(p), X.T@y,assume_a='pos')
        gcv_ridge = crit(lam_opt)
    return beta, gcv_ridge, lam_opt


# =========================
# CLASSICAL LIU
# =========================
def classical_liu(X, y, params_optimal = None, lambda_bounds = (1e-6,1e6), opt_method = 'SLSQP', 
                  criteria = 'GCV', log_change_of_variables=None):
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
            or 'CV' to use leave-one-out (LOO) Cross Valiation

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
        
    def gcv(params):
        lam, d = params
        lam = exp(lam)
        A = G + lam*jnp.eye(p)
        (c,lower)  = jsla.cho_factor(A)
        beta = jsla.cho_solve((c,lower), Xy + d*lam*beta_OLS)
        yhat = X @ beta
        P    = X @ jsla.cho_solve( (c,lower), X.T + d*lam*P_OLS )
        traceP = jnp.trace(P)
        return jnp.mean((jnp.asarray(y)-yhat)**2)/((1-traceP/n)**2)

    def cv(params):
        lam, d = params
        lam = exp(lam)
        A = G + lam*jnp.eye(p)
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
        res = opt.minimize(crit, np.array([log(1.0),0.5]), jac=grad,
                          bounds=[lambda_bounds,(0,1)], method=opt_method)

        lam_opt, d_opt = res.x
        lam_opt = exp(lam_opt)
        gcv_liu = res.fun
    else:
        lam_opt, d_opt = params_optimal
        gcv_liu = crit(params_optimal)

    beta = sla.solve(X.T@X + lam_opt*np.eye(p),
                     X.T@y + d_opt*lam_opt*np.array(beta_OLS),
                     assume_a = 'pos')

    return beta, gcv_liu, lam_opt, d_opt

# =========================
# CARDOT'S ESTIMATOR
# =========================
def cardot(X, y, R, params_optimal=None, lambda_bounds = (1e-6,1e6), opt_method = 'SLSQP', 
           criteria = 'GCV', log_change_of_variables=None):
    """ Cardot's generalized ridge regression estimator
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
            or 'CV' to use leave-one-out (LOO) Cross Valiation

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
        res = opt.minimize(crit, np.array([log(1.0),0.5]), jac=grad,
                          bounds=[lambda_bounds,(0,1)], method=opt_method)

        lam_opt, alpha_opt = res.x
        lam_opt = exp(lam_opt)
        gcv_cardot = res.fun
    else:
        lam_opt, alpha_opt = params_optimal
        gcv_cardot = crit(params_optimal)

    Q = lam_opt*(alpha_opt*np.eye(p) + (1-alpha_opt)*R)
    beta = sla.solve(X.T@X + Q, X.T@y, assume_a = 'pos')

    return beta, gcv_cardot, lam_opt, alpha_opt


# =========================
# FUNCTIONAL LIU
# =========================
def functional_liu(X, y, R, params_optimal=None, lambda_bounds = (1e-6,1e6), 
                   opt_method = 'SLSQP', criteria = 'GCV', log_change_of_variables=None):
    """ Functional Liu estimator: combination of Cardot's generalized ridge with Liu's biased estimator
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
            or 'CV' to use leave-one-out (LOO) Cross Valiation

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
        res = opt.minimize(crit, np.array([log(1.0),0.5,0.5]), jac=grad,
                          bounds=[lambda_bounds,(0,1),(0,1)], method=opt_method)

        lam_opt, d_opt, alpha_opt = res.x
        lam_opt = exp(lam_opt)
        gcv_fliu = res.fun
    else:
        lam_opt, d_opt, alpha_opt = params_optimal
        gcv_fliu = crit(params_optimal)

    Q = lam_opt*(alpha_opt*np.eye(p) + (1-alpha_opt)*R)
    beta = sla.solve(X.T@X + Q,
                     X.T@y + d_opt*(Q@np.linalg.lstsq(X,y,rcond=None)[0]),
                     assume_a = 'pos')

    return beta, gcv_fliu, lam_opt, d_opt, alpha_opt
