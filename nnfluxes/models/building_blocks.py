from jax import grad, vmap, random, jit
from jax import numpy as jnp
import jax

def MLP(layers):
    def activation(x):
        return jnp.tanh(x)
        #return jnp.maximum(x, 0)
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            b = jnp.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs
    return init, apply

def MLPDropout(layers, final_nonlin=False):
    if final_nonlin:
        final_activation = jax.nn.softplus
    else:
        final_activation = lambda x: x
    
    def activation(x):
        return jnp.tanh(x)
        #return jnp.relu(x)
        #return jnp.maximum(x, 0)
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            b = jnp.zeros(d_out)
            return W, b
            
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        
        return params
    def dropout(inputs, p, rng_key):
        k1, k2 = random.split(rng_key)
        return inputs * random.bernoulli(k1, p=1.0-p, shape=[inputs.shape[0]]) * (1.0/(1-p))
    def apply(params, inputs, p, rng_key):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            outputs = activation(outputs)
            inputs = dropout(outputs, p, rng_key)
        W, b = params[-1]
        outputs = final_activation(jnp.dot(inputs, W) + b)
        return outputs
    def apply_eval(params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = final_activation(jnp.dot(inputs, W) + b)
        return outputs
    return init, apply, apply_eval

