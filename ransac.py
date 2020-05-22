import numpy as np
import random as randn
import random as randint
# true values
_a = 0.5
_b = 0.3
x = 0
# samples
# samples
points = np.array([[x, _a * x + _b + .1 * np.random.randn() + (np.random.randint(100) == 0)

def leastSquare(data):
    # Simulated Annealing
    tau = 100
    bestfit = None
    besterr = float('inf')
    model = np.zeros(2)
    while tau >= 0.0001:
        for _ in range(10):
            grad = errorGrad(model, data)
            grad /= np.linalg.norm(grad)
            grad *= -1
            model += grad * tau

        tau *= 0.1
    return model


def ransac(data,
        # parameters for RANSAC
        n = 2, # required sample num to decide parameter
        k = 100, # max loop num
        t = 2.0, # threshold error val for inlier
        d = 800 # requrired inlier sample num to be correnct param
    ):

    good_models = []
    good_model_errors = []
    iterations = 0
    while iterations < k:
        sample = data[np.random.choice(len(data), 2, False)]
        param = getParamWithSamples(sample)

        inliers = []
        for p in data:
            if (p == sample).all(1).any(): continue
            if getError(param, p) > t:
                continue
            else:
                inliers.append(p)


        if len(inliers) > d:
            current_error = getModelError(param, data)
            good_models.append(param)
            good_model_errors.append(current_error)

        iterations += 1

    best_index = np.argmin(good_model_errors)
    return good_models[best_index]

a, b = ransac(data)
