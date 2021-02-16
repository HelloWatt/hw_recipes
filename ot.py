import numpy as np
from cvxpy import Variable, Parameter, Minimize, Problem, entr, ECOS_BB
from cvxpy import multiply as cvx_mul
from cvxpy import sum as cvx_sum


def compute_distances_alt(xdim, ydim):
    """
        :returns distance matrix for the grid on [0, 1) x [0, 1) with steps 1/xdim and 1/ydim
        with periodic boundary conditions
    :param xdim:
    :param ydim:
    :return:
    """

    xs = np.arange(xdim) / xdim
    ys = np.arange(ydim) / ydim
    half = 0.5
    x_tilde = half - np.abs(xs - half)
    y_tilde = half - np.abs(ys - half)
    dist = np.repeat(x_tilde[:, np.newaxis], ydim, 1) + np.repeat(y_tilde[np.newaxis, :], xdim, 0)
    return dist


def compute_joint(xpdf, ypdf, solver=ECOS_BB, solver_options={}, corr=0.0, gamma=0.0):
    """

    :param xpdf:
    :param ypdf:
    :param solver:
    :param solver_options:
    :param corr:
    :param gamma:
    :return:
    """

    xdim = xpdf.shape[0]
    ydim = ypdf.shape[0]
    xpdf = xpdf / xpdf.sum()
    ypdf = ypdf / ypdf.sum()

    dist = compute_distances_alt(xdim, ydim)

    plan = Variable((xdim, ydim))

    mx_trans = xpdf.reshape(-1, 1) * dist
    mu_trans_param = Parameter(mx_trans.shape, value=mx_trans)

    entropy = entr(plan)
    obj = Minimize(cvx_sum(cvx_mul(plan, mu_trans_param) - gamma*entropy))

    plan_i = cvx_sum(plan, axis=1)
    marginal_constraint = xpdf @ plan

    xs = np.arange(xdim)
    ys = np.arange(ydim)
    mean_x = np.sum(xs*xpdf)
    mean_y = np.sum(ys*ypdf)
    var_x = np.sum((xs - mean_x)**2*xpdf)
    var_y = np.sum((ys - mean_y)**2*ypdf)
    corr_rhs = (var_x * var_y) ** 0.5 * corr + mean_x * mean_y

    xx = np.tile(xpdf*xs, (ys.shape[0], 1)).T
    yy = np.tile(ys, (xs.shape[0], 1))

    corr_constraint = cvx_sum(cvx_mul(cvx_mul(xx, plan), yy))

    constraints = [marginal_constraint == ypdf,
                   corr_constraint == corr_rhs,
                   plan >= 0, plan <= 1,
                   plan_i == np.ones(xdim),
                   ]
    problem = Problem(obj, constraints)
    wd = problem.solve(solver=solver, **solver_options)
    jpdf = (xpdf * problem.variables()[0].value.T).T
    return jpdf

