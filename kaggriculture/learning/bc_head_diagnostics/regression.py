"""Small synthetic regressions for the analytic diagnostic contract."""
import json
import numpy as np
from .math import objective, direction_stats, first_adam, collision_summary, nearest


def run():
    rng = np.random.default_rng(7)
    h = rng.normal(size=(9, 4)); theta = rng.normal(size=(4, 3))
    y = np.arange(9) % 3; weight = np.arange(1, 10, dtype=float)
    before = theta.copy(); theta.flags.writeable = False
    loss, grad, p = objective(h, y, theta, weight)
    epsilon = 1e-5; numerical = np.empty_like(grad)
    for index in np.ndindex(theta.shape):
        delta = np.zeros_like(theta); delta[index] = epsilon
        numerical[index] = (objective(h,y,theta+delta,weight)[0]-objective(h,y,theta-delta,weight)[0])/(2*epsilon)
    assert np.allclose(grad,numerical,rtol=1e-7,atol=1e-9)
    direction = rng.normal(size=theta.shape)
    stats = direction_stats(h,weight,p,grad,direction)
    numerical_curve = np.sum(direction*(objective(h,y,theta+epsilon*direction,weight)[1]-objective(h,y,theta-epsilon*direction,weight)[1]))/(2*epsilon)
    assert np.isclose(stats['curvature'],numerical_curve,rtol=1e-7,atol=1e-9)
    assert stats['curvature'] >= 0
    common_shift = np.repeat(rng.normal(size=(4,1)),3,axis=1)
    shift = direction_stats(h,weight,p,grad,common_shift)
    assert abs(shift['derivative']) < 1e-12 and shift['curvature'] < 1e-25
    padded = objective(np.vstack([h,h[:2]]),np.r_[y,y[:2]],theta,np.r_[weight,0,0])
    assert np.isclose(loss,padded[0]) and np.allclose(grad,padded[1])
    g = np.array([-3.,-1e-10,0.,1e-10,2.]); lr=.0003; eps=1e-8
    m=(1-.9)*g; v=(1-.999)*g*g
    explicit=-lr*(m/(1-.9))/(np.sqrt(v/(1-.999))+eps)
    assert np.allclose(first_adam(g,lr,eps),explicit,rtol=1e-14,atol=1e-20)
    assert collision_summary(np.array([[0.,1.],[-0.,1.],[2.,1.]]),[0,1,1]) == dict(unique_hidden=2,different_label_groups=1)
    assert nearest(np.zeros(2),np.array([[1.,0.],[-1.,0.]]),['b','a'],np.array([True,True])) == (1,1.)
    assert nearest(np.zeros(2),np.zeros((1,2)),['a'],np.array([False])) is None
    assert np.array_equal(theta,before)
    return dict(passed=True,checks=8,synthetic_only=True,real_parameter_updates=0)


if __name__ == '__main__': print(json.dumps(run()))
