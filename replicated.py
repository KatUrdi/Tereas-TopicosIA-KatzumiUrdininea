from fastapi import FastAPI
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import stats

app = FastAPI()

def create_model(name, fields):
    return type(name, (BaseModel,), fields)

OptimizationRequest = create_model('OptimizationRequest', {'initial_value': (float,)})
IntegrationRequest = create_model('IntegrationRequest', {'lower_limit': (float,), 'upper_limit': (float,)})
StatisticsRequest = create_model('StatisticsRequest', {'data': (list[float],)})

def objective_function(x):
    """
    Quadratic objective function: f(x) = x^2 + 5x + 10.
    """
    return x**2 + 5 * x + 10

@app.post('/optimize/')
def optimize(request: OptimizationRequest):
    """
    Endpoint to optimize a quadratic objective function.
    """
    result = minimize(objective_function, request.initial_value)
    return {'optimal_value': result.x.tolist()}

def integrand_function(x):
    """
    Function to integrate: f(x) = x^2.
    """
    return x**2

@app.post('/integrate/')
def integrate(request: IntegrationRequest):
    """
    Endpoint to calculate the area under the curve of the function f(x) = x^2
    between the specified lower and upper limits using numerical integration.
    """
    area, error = quad(integrand_function, request.lower_limit, request.upper_limit)
    return {'area_under_curve': area, 'error_estimate': error}

def calculate_statistics(data):
    """
    Calculate the mean and variance of the given data.
    """
    mean = stats.tmean(data)
    variance = stats.tvar(data)
    return mean, variance

@app.post('/statistics/')
def statistics(request: StatisticsRequest):
    """
    Endpoint to calculate the mean and variance of a given list of numbers.
    """
    mean, variance = calculate_statistics(request.data)
    return {'mean': mean, 'variance': variance}

