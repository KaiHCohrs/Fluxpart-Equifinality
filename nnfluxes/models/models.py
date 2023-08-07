import sys
import os
import itertools
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as orandom

import doubleml as dml
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor

import jax
from jax import grad, vmap, random, jit
from jax import numpy as jnp
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import optimizer, make_schedule, l2_norm
from jax.lax import cond

from tqdm import trange
import optax

import torch.nn as nn
import torch
import torch.nn.functional as torchf
import numpy as np
import sys

from .building_blocks import MLP, MLPDropout#, RespirationModel

class BaseModel:
    def __init__(self, nn_1, nn_2, ensemble_size, p=0.01, weight_decay=0, rng_key=random.PRNGKey(0)):
        self.init1, self.apply1, self.apply_eval1 = MLPDropout(nn_1)
        self.init2, self.apply2, self.apply_eval2 = MLPDropout(nn_2)
        self.ensemble_size = ensemble_size
        self.p = p
        
        # Random keys
        rng_key1, rng_key2, rng_key = random.split(rng_key, 3)
        
        keys_1 = random.split(rng_key1, ensemble_size)
        keys_2 = random.split(rng_key2, ensemble_size)
        
        # Initialize
        self.params1 = vmap(self.init1)(keys_1)
        self.params2 = vmap(self.init2)(keys_2)

        
        self.schedule1 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer1 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule1,
                                        weight_decay=weight_decay,
                                        ),
                            )
        self.opt_state1 = vmap(self.optimizer1.init)(self.params1)


        self.schedule2 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer2 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule2,
                                        weight_decay=0)
                            )
        self.opt_state2 = vmap(self.optimizer2.init)(self.params2)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.Q10_log = []
        # I need to compute it myself
        self.lr_log = []
        self.best_i = []
        self.Q10_log = [jnp.array(self.ensemble_size*[0])]
        self.loss_test_log = [jnp.array(self.ensemble_size*[jnp.inf])]
        
        
    # Define the forward pass
    def net_forward(self, params1, params2, inputs1, inputs2, inputs3, inputs4, p, rng_key):
        rng_key1, rng_key2 = random.split(rng_key)
        
        GPP = jax.nn.softplus(self.apply1(params1, inputs1, p, rng_key1))
        RECO = jax.nn.softplus(self.apply2(params2, inputs2, p, rng_key2))

        Y_pred = -GPP + RECO
        return Y_pred

    def net_forward_test(self, params1, params2, inputs1, inputs2, inputs3, inputs4):
        
        GPP = jax.nn.softplus(self.apply_eval1(params1, inputs1))
        RECO = jax.nn.softplus(self.apply_eval2(params2, inputs2))

        Y_pred = -GPP + RECO
        return Y_pred, GPP, RECO, None


    # loss functions
    def loss(self, params1, params2, batch, p, rng_key):
        inputs1, inputs2, inputs3, inputs4, targets = batch
        
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0, 0, 0, 0, None, 0))(params1, params2, 
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4,
                                                                            p, rng_key)
        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss

    def loss_test(self, params1, params2, batch):
        inputs1, inputs2, inputs3, inputs4, targets = batch

        # Compute forward pass
        outputs, _, _, _ = vmap(self.net_forward_test, (None, None, 0, 0, 0, 0))(params1, params2, 
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4)

        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss


    # monitor loss functions
    def monitor_loss(self, params1, params2, batch, p, rng_key):
        loss_value = self.loss(params1, params2, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params1, params2, batch):
        loss_value = self.loss_test(params1, params2, batch)
        return loss_value

    # Define the update step
    def step(self, i, params1, params2, opt_state1, opt_state2, batch, p, rng_key):
        grads1 = jax.grad(self.loss, argnums=0)(params1, params2, batch, p, rng_key)
        grads2 = jax.grad(self.loss, argnums=1)(params1, params2, batch, p, rng_key)
        updates1, opt_state1 = self.optimizer1.update(grads1, opt_state1, params1) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates2, opt_state2 = self.optimizer2.update(grads2, opt_state2, params2) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        
        params1 = optax.apply_updates(params1, updates1)
        params2 = optax.apply_updates(params2, updates2)

        return params1, params2, opt_state1, opt_state2    
    
    def update_weights(self, params1, params2, params1_best, params2_best):
        return params1, params2
        
    def keep_weights(self, params1, params2, params1_best, params2_best):
        return params1_best, params2_best
    
    def model_selection(self, update, params1, params2, params1_best, params2_best):
        return jax.lax.cond(update, self.update_weights, self.keep_weights, params1, params2, params1_best, params2_best)
    
    def batch_normalize(self, data_val, norm_const):
        X1, X2, X3, X4, y = data_val

        (mu_X1, sigma_X1), (mu_X2, sigma_X2), (mu_X3, sigma_X3), (mu_X4, sigma_X4), (mu_y, sigma_y) = norm_const
        X1 = (X1 - mu_X1)/sigma_X1
        X2 = (X2 - mu_X2)/sigma_X2
        #X3 = (X3 - mu_X3)/sigma_X3
        #X4 = (X4 - mu_X4)/sigma_X4
        y = (y - mu_y)/sigma_y
        
        return [X1, X2, X3, X4, y]
        
    # Optimize parameters in a loop
    def fit(self, dataset, data_train, data_val, data_test, rng_key, nIter = 1000):
        self.params1_best = self.params1
        self.params2_best = self.params2

        data = iter(dataset)
        (self.mu_X1, self.sigma_X1), \
        (self.mu_X2, self.sigma_X2), \
        (self.mu_X3, self.sigma_X3), \
        (self.mu_X4, self.sigma_X4), \
        (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)

        # Vectorize along 0 axis which is ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0, None, 0)))   # vmap(step, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, None, 0))) # vmap(monitor_loss, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes = (0, 0, 0))) # vmap(monitor_loss_test, ensemble) -> loss_test -> vmap(net_forward_test, batch) (makes sense)
        
        v_model_selection = jit(vmap(self.model_selection, in_axes = (0, 0, 0, 0, 0))) # Completely ensemblewise, jitable?
        v_batch_normalize = vmap(self.batch_normalize, in_axes = (None, 0)) # Maybe dont jit because we apply it once

        data_train = v_batch_normalize(data_train, dataset.norm_const)
        data_val = v_batch_normalize(data_val, dataset.norm_const)
        data_test = v_batch_normalize(data_test, dataset.norm_const)

        # Main training loop
        for it in pbar:
            batch = next(data)
            rng_key, rng_key_ens_batch = random.split(rng_key, 2)
            rng_key_ens_batch = random.split(rng_key_ens_batch, batch[-1].shape[0]*batch[-1].shape[1]).reshape(batch[-1].shape[0], batch[-1].shape[1], 2)
            self.params1, self.params2, self.opt_state1, self.opt_state2 = v_step(it, self.params1, self.params2, 
                                                                        self.opt_state1, self.opt_state2,
                                                                        batch, self.p, jnp.array(rng_key_ens_batch))
            
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10_log[0])
                loss_value = v_monitor_loss(self.params1, self.params2, batch, self.p, jnp.array(rng_key_ens_batch))
                self.loss_log.append(loss_value)
                
                loss_test_value = v_monitor_loss_test(self.params1, self.params2, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                # Compute Values for logging
                loss_train_value = v_monitor_loss_test(self.params1, self.params2, data_train)
                self.train_log.append(loss_train_value)

                loss_val_value = v_monitor_loss_test(self.params1, self.params2, data_val)
                self.val_log.append(loss_val_value)

                loss_test_value = v_monitor_loss_test(self.params1, self.params2, data_test)
                self.test_log.append(loss_test_value)

                pbar.set_postfix({'Max loss': loss_value.max(), 'Max test loss': loss_test_value.max()})

                self.params1_best, self.params2_best = v_model_selection(update, self.params1, self.params2, self.params1_best, self.params2_best)
                
    # Evaluates predictions at test points
    def posterior(self, x1, x2, x3, x4):
        normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
        denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

        # This is necessary so that for each network we get slightly different inputs given their normalizing factors
        x1 = jnp.tile(x1[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x2 = jnp.tile(x2[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x3 = jnp.tile(x3[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x4 = jnp.tile(x4[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        
        inputs1 = normalize(x1, self.mu_X1, self.sigma_X1)
        inputs2 = normalize(x2, self.mu_X2, self.sigma_X2)
        inputs3 = x3
        inputs4 = x4
        
        # For that reason we also vectorize over the ensemble axis for all
        samples, samples_GPP, samples_RECO, _ = vmap(self.net_forward_test, (0, 0, 0, 0, 0, 0))(self.params1_best, self.params2_best, inputs1, inputs2, inputs3, inputs4)
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_GPP = denormalize(samples_GPP, self.mu_y, self.sigma_y)
        samples_RECO = denormalize(samples_RECO, self.mu_y, self.sigma_y)
        
        return samples, samples_GPP, samples_RECO, None


class Nightconstrained:
    def __init__(self, nn_1, nn_2, ensemble_size, p=0.01, weight_decay=0, rng_key=random.PRNGKey(0)):
        self.init1, self.apply1, self.apply_eval1 = MLPDropout(nn_1)
        self.init2, self.apply2, self.apply_eval2 = MLPDropout(nn_2)
        self.ensemble_size = ensemble_size
        self.p = p
        
        # Random keys
        rng_key1, rng_key2, rng_key = random.split(rng_key, 3)
        
        keys_1 = random.split(rng_key1, ensemble_size)
        keys_2 = random.split(rng_key2, ensemble_size)
        
        # Initialize
        self.params1 = vmap(self.init1)(keys_1)
        self.params2 = vmap(self.init2)(keys_2)

        
        self.schedule1 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer1 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule1,
                                        weight_decay=weight_decay,
                                        ),
                            )
        self.opt_state1 = vmap(self.optimizer1.init)(self.params1)


        self.schedule2 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer2 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule2,
                                        weight_decay=0)
                            )
        self.opt_state2 = vmap(self.optimizer2.init)(self.params2)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.Q10_log = [jnp.array(self.ensemble_size*[0])]
        self.loss_test_log = [jnp.array(self.ensemble_size*[jnp.inf])]
        
        
    # Define the forward pass
    def net_forward(self, params1, params2, inputs1, inputs2, inputs3, inputs4, p, rng_key):
        rng_key1, rng_key2 = random.split(rng_key)
        
        GPP = jax.nn.softplus(self.apply1(params1, inputs1, p, rng_key1))*jnp.abs(inputs3-1)
        RECO = jax.nn.softplus(self.apply2(params2, inputs2, p, rng_key2))

        Y_pred = -GPP + RECO
        return Y_pred

    def net_forward_test(self, params1, params2, inputs1, inputs2, inputs3, inputs4):
        
        GPP = jax.nn.softplus(self.apply_eval1(params1, inputs1))*jnp.abs(inputs3-1)
        RECO = jax.nn.softplus(self.apply_eval2(params2, inputs2))

        Y_pred = -GPP + RECO
        return Y_pred, GPP, RECO, None


    # loss functions
    def loss(self, params1, params2, batch, p, rng_key):
        inputs1, inputs2, inputs3, inputs4, targets = batch
        
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0, 0, 0, 0, None, 0))(params1, params2, 
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4,
                                                                            p, rng_key)
        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss

    def loss_test(self, params1, params2, batch):
        inputs1, inputs2, inputs3, inputs4, targets = batch

        # Compute forward pass
        outputs, _, _, _ = vmap(self.net_forward_test, (None, None, 0, 0, 0, 0))(params1, params2, 
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4)

        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss


    # monitor loss functions
    def monitor_loss(self, params1, params2, batch, p, rng_key):
        loss_value = self.loss(params1, params2, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params1, params2, batch):
        loss_value = self.loss_test(params1, params2, batch)
        return loss_value

    # Define the update step
    def step(self, i, params1, params2, opt_state1, opt_state2, batch, p, rng_key):
        grads1 = jax.grad(self.loss, argnums=0)(params1, params2, batch, p, rng_key)
        grads2 = jax.grad(self.loss, argnums=1)(params1, params2, batch, p, rng_key)
        updates1, opt_state1 = self.optimizer1.update(grads1, opt_state1, params1) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates2, opt_state2 = self.optimizer2.update(grads2, opt_state2, params2) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        
        params1 = optax.apply_updates(params1, updates1)
        params2 = optax.apply_updates(params2, updates2)

        return params1, params2, opt_state1, opt_state2    
    
    def update_weights(self, params1, params2, params1_best, params2_best):
        return params1, params2
        
    def keep_weights(self, params1, params2, params1_best, params2_best):
        return params1_best, params2_best
    
    def model_selection(self, update, params1, params2, params1_best, params2_best):
        return jax.lax.cond(update, self.update_weights, self.keep_weights, params1, params2, params1_best, params2_best)
    
    def batch_normalize(self, data_val, norm_const):
        X1, X2, X3, X4, y = data_val

        (mu_X1, sigma_X1), (mu_X2, sigma_X2), (mu_X3, sigma_X3), (mu_X4, sigma_X4), (mu_y, sigma_y) = norm_const
        X1 = (X1 - mu_X1)/sigma_X1
        X2 = (X2 - mu_X2)/sigma_X2
        #X3 = (X3 - mu_X3)/sigma_X3
        #X4 = (X4 - mu_X4)/sigma_X4
        y = (y - mu_y)/sigma_y
        
        return [X1, X2, X3, X4, y]
        
    # Optimize parameters in a loop
    def fit(self, dataset, data_train, data_val, data_test, rng_key, nIter = 1000):
        self.params1_best = self.params1
        self.params2_best = self.params2

        data = iter(dataset)
        (self.mu_X1, self.sigma_X1), \
        (self.mu_X2, self.sigma_X2), \
        (self.mu_X3, self.sigma_X3), \
        (self.mu_X4, self.sigma_X4), \
        (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)

        # Vectorize along 0 axis which is ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0, None, 0)))   # vmap(step, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, None, 0))) # vmap(monitor_loss, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes = (0, 0, 0))) # vmap(monitor_loss_test, ensemble) -> loss_test -> vmap(net_forward_test, batch) (makes sense)
        
        v_model_selection = jit(vmap(self.model_selection, in_axes = (0, 0, 0, 0, 0))) # Completely ensemblewise, jitable?
        v_batch_normalize = vmap(self.batch_normalize, in_axes = (None, 0)) # Maybe dont jit because we apply it once

        data_train = v_batch_normalize(data_train, dataset.norm_const)
        data_val = v_batch_normalize(data_val, dataset.norm_const)
        data_test = v_batch_normalize(data_test, dataset.norm_const)

        # Main training loop
        for it in pbar:
            batch = next(data)
            rng_key, rng_key_ens_batch = random.split(rng_key, 2)
            rng_key_ens_batch = random.split(rng_key_ens_batch, batch[-1].shape[0]*batch[-1].shape[1]).reshape(batch[-1].shape[0], batch[-1].shape[1], 2)
            self.params1, self.params2, self.opt_state1, self.opt_state2 = v_step(it, self.params1, self.params2, 
                                                                        self.opt_state1, self.opt_state2,
                                                                        batch, self.p, jnp.array(rng_key_ens_batch))
            
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10_log[0])
                loss_value = v_monitor_loss(self.params1, self.params2, batch, self.p, jnp.array(rng_key_ens_batch))
                self.loss_log.append(loss_value)
                
                loss_test_value = v_monitor_loss_test(self.params1, self.params2, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                # Compute Values for logging
                loss_train_value = v_monitor_loss_test(self.params1, self.params2, data_train)
                self.train_log.append(loss_train_value)

                loss_val_value = v_monitor_loss_test(self.params1, self.params2, data_val)
                self.val_log.append(loss_val_value)

                loss_test_value = v_monitor_loss_test(self.params1, self.params2, data_test)
                self.test_log.append(loss_test_value)

                pbar.set_postfix({'Max loss': loss_value.max(), 'Max test loss': loss_test_value.max()})

                self.params1_best, self.params2_best = v_model_selection(update, self.params1, self.params2, self.params1_best, self.params2_best)
                
    # Evaluates predictions at test points
    def posterior(self, x1, x2, x3, x4):
        normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
        denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

        # This is necessary so that for each network we get slightly different inputs given their normalizing factors
        x1 = jnp.tile(x1[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x2 = jnp.tile(x2[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x3 = jnp.tile(x3[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x4 = jnp.tile(x4[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        
        inputs1 = normalize(x1, self.mu_X1, self.sigma_X1)
        inputs2 = normalize(x2, self.mu_X2, self.sigma_X2)
        inputs3 = x3
        inputs4 = x4
        
        # For that reason we also vectorize over the ensemble axis for all
        samples, samples_GPP, samples_RECO, _ = vmap(self.net_forward_test, (0, 0, 0, 0, 0, 0))(self.params1_best, self.params2_best, inputs1, inputs2, inputs3, inputs4)
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_GPP = denormalize(samples_GPP, self.mu_y, self.sigma_y)
        samples_RECO = denormalize(samples_RECO, self.mu_y, self.sigma_y)
        
        return samples, samples_GPP, samples_RECO, None
    
    

class LUE:
    def __init__(self, nn_1, nn_2, ensemble_size, p=0.01, weight_decay=0, rng_key=random.PRNGKey(0)):
        self.init1, self.apply1, self.apply_eval1 = MLPDropout(nn_1)
        self.init2, self.apply2, self.apply_eval2 = MLPDropout(nn_2)
        self.ensemble_size = ensemble_size
        self.p = p
        
        # Random keys
        rng_key1, rng_key2, rng_key = random.split(rng_key, 3)
        
        keys_1 = random.split(rng_key1, ensemble_size)
        keys_2 = random.split(rng_key2, ensemble_size)
        
        # Initialize
        self.params1 = vmap(self.init1)(keys_1)
        self.params2 = vmap(self.init2)(keys_2)

        
        self.schedule1 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer1 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule1,
                                        weight_decay=weight_decay,
                                        ),
                            )
        self.opt_state1 = vmap(self.optimizer1.init)(self.params1)


        self.schedule2 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer2 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule2,
                                        weight_decay=0)
                            )
        self.opt_state2 = vmap(self.optimizer2.init)(self.params2)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.Q10_log = [jnp.array(self.ensemble_size*[0])]
        self.loss_test_log = [jnp.array(self.ensemble_size*[jnp.inf])]
        
        
    # Define the forward pass
    def net_forward(self, params1, params2, inputs1, inputs2, inputs3, inputs4, p, rng_key):
        rng_key1, rng_key2 = random.split(rng_key)
        
        GPP = jax.nn.softplus(self.apply1(params1, inputs1, p, rng_key1))*inputs3
        RECO = jax.nn.softplus(self.apply2(params2, inputs2, p, rng_key2))

        Y_pred = -GPP + RECO
        return Y_pred

    def net_forward_test(self, params1, params2, inputs1, inputs2, inputs3, inputs4):
        
        LUE = jax.nn.softplus(self.apply_eval1(params1, inputs1))
        GPP = LUE*inputs3
        RECO = jax.nn.softplus(self.apply_eval2(params2, inputs2))

        Y_pred = -GPP + RECO
        return Y_pred, GPP, RECO, LUE


    # loss functions
    def loss(self, params1, params2, batch, p, rng_key):
        inputs1, inputs2, inputs3, inputs4, targets = batch
        
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0, 0, 0, 0, None, 0))(params1, params2, 
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4,
                                                                            p, rng_key)
        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss

    def loss_test(self, params1, params2, batch):
        inputs1, inputs2, inputs3, inputs4, targets = batch

        # Compute forward pass
        outputs, _, _, _ = vmap(self.net_forward_test, (None, None, 0, 0, 0, 0))(params1, params2, 
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4)

        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss


    # monitor loss functions
    def monitor_loss(self, params1, params2, batch, p, rng_key):
        loss_value = self.loss(params1, params2, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params1, params2, batch):
        loss_value = self.loss_test(params1, params2, batch)
        return loss_value

    # Define the update step
    def step(self, i, params1, params2, opt_state1, opt_state2, batch, p, rng_key):
        grads1 = jax.grad(self.loss, argnums=0)(params1, params2, batch, p, rng_key)
        grads2 = jax.grad(self.loss, argnums=1)(params1, params2, batch, p, rng_key)
        updates1, opt_state1 = self.optimizer1.update(grads1, opt_state1, params1) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates2, opt_state2 = self.optimizer2.update(grads2, opt_state2, params2) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        
        params1 = optax.apply_updates(params1, updates1)
        params2 = optax.apply_updates(params2, updates2)

        return params1, params2, opt_state1, opt_state2    
    
    def update_weights(self, params1, params2, params1_best, params2_best):
        return params1, params2
        
    def keep_weights(self, params1, params2, params1_best, params2_best):
        return params1_best, params2_best
    
    def model_selection(self, update, params1, params2, params1_best, params2_best):
        return jax.lax.cond(update, self.update_weights, self.keep_weights, params1, params2, params1_best, params2_best)
    
    def batch_normalize(self, data_val, norm_const):
        X1, X2, X3, X4, y = data_val

        (mu_X1, sigma_X1), (mu_X2, sigma_X2), (mu_X3, sigma_X3), (mu_X4, sigma_X4), (mu_y, sigma_y) = norm_const
        X1 = (X1 - mu_X1)/sigma_X1
        X2 = (X2 - mu_X2)/sigma_X2
        X3 = (X3 - mu_X3)/sigma_X3
        #X4 = (X4 - mu_X4)/sigma_X4
        y = (y - mu_y)/sigma_y
        
        return [X1, X2, X3, X4, y]
        
    # Optimize parameters in a loop
    def fit(self, dataset, data_train, data_val, data_test, rng_key, nIter = 1000):
        self.params1_best = self.params1
        self.params2_best = self.params2

        data = iter(dataset)
        (self.mu_X1, self.sigma_X1), \
        (self.mu_X2, self.sigma_X2), \
        (self.mu_X3, self.sigma_X3), \
        (self.mu_X4, self.sigma_X4), \
        (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)

        # Vectorize along 0 axis which is ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0, None, 0)))   # vmap(step, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, None, 0))) # vmap(monitor_loss, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes = (0, 0, 0))) # vmap(monitor_loss_test, ensemble) -> loss_test -> vmap(net_forward_test, batch) (makes sense)
        
        v_model_selection = jit(vmap(self.model_selection, in_axes = (0, 0, 0, 0, 0))) # Completely ensemblewise, jitable?
        v_batch_normalize = vmap(self.batch_normalize, in_axes = (None, 0)) # Maybe dont jit because we apply it once

        data_train = v_batch_normalize(data_train, dataset.norm_const)
        data_val = v_batch_normalize(data_val, dataset.norm_const)
        data_test = v_batch_normalize(data_test, dataset.norm_const)

        # Main training loop
        for it in pbar:
            batch = next(data)
            rng_key, rng_key_ens_batch = random.split(rng_key, 2)
            rng_key_ens_batch = random.split(rng_key_ens_batch, batch[-1].shape[0]*batch[-1].shape[1]).reshape(batch[-1].shape[0], batch[-1].shape[1], 2)
            self.params1, self.params2, self.opt_state1, self.opt_state2 = v_step(it, self.params1, self.params2, 
                                                                        self.opt_state1, self.opt_state2,
                                                                        batch, self.p, jnp.array(rng_key_ens_batch))
            
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10_log[0])
                loss_value = v_monitor_loss(self.params1, self.params2, batch, self.p, jnp.array(rng_key_ens_batch))
                self.loss_log.append(loss_value)
                
                loss_test_value = v_monitor_loss_test(self.params1, self.params2, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                # Compute Values for logging
                loss_train_value = v_monitor_loss_test(self.params1, self.params2, data_train)
                self.train_log.append(loss_train_value)

                loss_val_value = v_monitor_loss_test(self.params1, self.params2, data_val)
                self.val_log.append(loss_val_value)

                loss_test_value = v_monitor_loss_test(self.params1, self.params2, data_test)
                self.test_log.append(loss_test_value)

                pbar.set_postfix({'Max loss': loss_value.max(), 'Max test loss': loss_test_value.max()})

                self.params1_best, self.params2_best = v_model_selection(update, self.params1, self.params2, self.params1_best, self.params2_best)
                
    # Evaluates predictions at test points
    def posterior(self, x1, x2, x3, x4):
        normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
        denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

        # This is necessary so that for each network we get slightly different inputs given their normalizing factors
        x1 = jnp.tile(x1[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x2 = jnp.tile(x2[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x3 = jnp.tile(x3[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x4 = jnp.tile(x4[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        
        inputs1 = normalize(x1, self.mu_X1, self.sigma_X1)
        inputs2 = normalize(x2, self.mu_X2, self.sigma_X2)
        inputs3 = normalize(x3, self.mu_X3, self.sigma_X3)
        inputs4 = x4
        
        # For that reason we also vectorize over the ensemble axis for all
        samples, samples_GPP, samples_RECO, samples_LUE = vmap(self.net_forward_test, (0, 0, 0, 0, 0, 0))(self.params1_best, self.params2_best, inputs1, inputs2, inputs3, inputs4)
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_GPP = denormalize(samples_GPP, self.mu_y, self.sigma_y)
        samples_RECO = denormalize(samples_RECO, self.mu_y, self.sigma_y)
        
        return samples, samples_GPP, samples_RECO, samples_LUE




class Q10:
    def __init__(self, nn_1, nn_2, ensemble_size, p=0.01, weight_decay=0, rng_key=random.PRNGKey(0), Q10_mean_guess=1.5):
        self.init1, self.apply1, self.apply_eval1 = MLPDropout(nn_1)
        self.init2, self.apply2, self.apply_eval2 = MLPDropout(nn_2)
        self.Q10_mean_guess = Q10_mean_guess
        self.Q10_std_guess = 0.1
        self.ensemble_size = ensemble_size
        self.p = p
        
        # Random keys
        rng_key1, rng_key2, rng_key3, rng_key = random.split(rng_key, 4)
        
        keys_1 = random.split(rng_key1, ensemble_size)
        keys_2 = random.split(rng_key2, ensemble_size)
        
        # Initialize
        self.params1 = vmap(self.init1)(keys_1)
        self.params2 = vmap(self.init2)(keys_2)
        self.Q10 = self.Q10_mean_guess + self.Q10_std_guess * random.normal(rng_key3, (self.ensemble_size,1))
        self.Q10_init = self.Q10

        
        self.schedule1 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer1 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule1,
                                        weight_decay=weight_decay,
                                        ),
                            )
        self.opt_state1 = vmap(self.optimizer1.init)(self.params1)


        self.schedule2 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer2 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule2,
                                        weight_decay=0)
                            )
        self.opt_state2 = vmap(self.optimizer2.init)(self.params2)


        self.schedule_Q10 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)

        self.optimizer_Q10 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule_Q10,
                                        weight_decay=0)
                            )

        self.opt_state_Q10 = vmap(self.optimizer_Q10.init)(self.Q10)
        #self.key_opt_state_Q10 = vmap(self.optimizer_Q10)(keys_4)


        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.Q10_log = []
        self.loss_test_log = [jnp.array(self.ensemble_size*[jnp.inf])]
        
        
    # Define the forward pass
    def net_forward(self, params1, params2, Q10, inputs1, inputs2, inputs3, inputs4, p, rng_key):
        rng_key1, rng_key2 = random.split(rng_key)
        
        GPP = jax.nn.softplus(self.apply1(params1, inputs1, p, rng_key1))
        Rb = jax.nn.softplus(self.apply2(params2, inputs2, p, rng_key2))
        RECO = Rb * Q10 ** (0.1 * (inputs4 - 15))

        Y_pred = -GPP + RECO
        return Y_pred

    def net_forward_test(self, params1, params2, Q10, inputs1, inputs2, inputs3, inputs4):
        
        GPP = jax.nn.softplus(self.apply_eval1(params1, inputs1))
        Rb = jax.nn.softplus(self.apply_eval2(params2, inputs2))
        RECO = Rb * Q10 ** (0.1 * (inputs4 - 15))

        Y_pred = -GPP + RECO
        return Y_pred, GPP, RECO, None


    # loss functions
    def loss(self, params1, params2, Q10, batch, p, rng_key):
        inputs1, inputs2, inputs3, inputs4, targets = batch
        
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, None, 0, 0, 0, 0, None, 0))(params1, params2, Q10,
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4,
                                                                            p, rng_key)
        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss

    def loss_test(self, params1, params2, Q10, batch):
        inputs1, inputs2, inputs3, inputs4, targets = batch

        # Compute forward pass
        outputs, _, _, _ = vmap(self.net_forward_test, (None, None, None, 0, 0, 0, 0))(params1, params2, Q10,
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4)

        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss


    # monitor loss functions
    def monitor_loss(self, params1, params2, Q10, batch, p, rng_key):
        loss_value = self.loss(params1, params2, Q10, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params1, params2, Q10, batch):
        loss_value = self.loss_test(params1, params2, Q10, batch)
        return loss_value

    # Define the update step
    def step(self, i, params1, params2, Q10, opt_state1, opt_state2, opt_state_Q10, batch, p, rng_key):
        grads1 = jax.grad(self.loss, argnums=0)(params1, params2, Q10, batch, p, rng_key)
        grads2 = jax.grad(self.loss, argnums=1)(params1, params2, Q10, batch, p, rng_key)
        grads_Q10 = jax.grad(self.loss, argnums=2)(params1, params2, Q10, batch, p, rng_key)
        updates1, opt_state1 = self.optimizer1.update(grads1, opt_state1, params1) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates2, opt_state2 = self.optimizer2.update(grads2, opt_state2, params2) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates_Q10, opt_state_Q10 = self.optimizer_Q10.update(grads_Q10, opt_state_Q10, Q10)
        
        params1 = optax.apply_updates(params1, updates1)
        params2 = optax.apply_updates(params2, updates2)
        Q10 = optax.apply_updates(Q10, updates_Q10)

        return params1, params2, Q10, opt_state1, opt_state2, opt_state_Q10
    
    def update_weights(self, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return params1, params2, Q10
        
    def keep_weights(self, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return params1_best, params2_best, Q10_best
    
    def model_selection(self, update, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return jax.lax.cond(update, self.update_weights, self.keep_weights, params1, params2, Q10, params1_best, params2_best, Q10_best)
    
    def batch_normalize(self, data_val, norm_const):
        X1, X2, X3, X4, y = data_val

        (mu_X1, sigma_X1), (mu_X2, sigma_X2), (mu_X3, sigma_X3), (mu_X4, sigma_X4), (mu_y, sigma_y) = norm_const
        X1 = (X1 - mu_X1)/sigma_X1
        X2 = (X2 - mu_X2)/sigma_X2
        #X3 = (X3 - mu_X3)/sigma_X3
        #X4 = (X4 - mu_X4)/sigma_X4
        y = (y - mu_y)/sigma_y
        
        return [X1, X2, X3, X4, y]
        
    # Optimize parameters in a loop
    def fit(self, dataset, data_train, data_val, data_test, rng_key, nIter = 1000):
        self.params1_best = self.params1
        self.params2_best = self.params2
        self.Q10_best = self.Q10

        data = iter(dataset)
        (self.mu_X1, self.sigma_X1), \
        (self.mu_X2, self.sigma_X2), \
        (self.mu_X3, self.sigma_X3), \
        (self.mu_X4, self.sigma_X4), \
        (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)

        # Vectorize along 0 axis which is ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0, 0, 0, None, 0)))   # vmap(step, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, 0, None, 0))) # vmap(monitor_loss, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes = (0, 0, 0, 0))) # vmap(monitor_loss_test, ensemble) -> loss_test -> vmap(net_forward_test, batch) (makes sense)
        
        v_model_selection = jit(vmap(self.model_selection, in_axes = (0, 0, 0, 0, 0, 0, 0))) # Completely ensemblewise, jitable?
        v_batch_normalize = vmap(self.batch_normalize, in_axes = (None, 0)) # Maybe dont jit because we apply it once

        data_train = v_batch_normalize(data_train, dataset.norm_const)
        data_val = v_batch_normalize(data_val, dataset.norm_const)
        data_test = v_batch_normalize(data_test, dataset.norm_const)

        # Main training loop
        for it in pbar:
            batch = next(data)
            rng_key, rng_key_ens_batch = random.split(rng_key, 2)
            rng_key_ens_batch = random.split(rng_key_ens_batch, batch[-1].shape[0]*batch[-1].shape[1]).reshape(batch[-1].shape[0], batch[-1].shape[1], 2)
            self.params1, self.params2, self.Q10, self.opt_state1, self.opt_state2, self.opt_state_Q10 = v_step(it, self.params1, self.params2, self.Q10,
                                                                        self.opt_state1, self.opt_state2, self.opt_state_Q10,
                                                                        batch, self.p, jnp.array(rng_key_ens_batch))
            
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10[:,0])
                loss_value = v_monitor_loss(self.params1, self.params2, self.Q10, batch, self.p, jnp.array(rng_key_ens_batch))
                self.loss_log.append(loss_value)
                
                loss_test_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                # Compute Values for logging
                loss_train_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_train)
                self.train_log.append(loss_train_value)

                loss_val_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_val)
                self.val_log.append(loss_val_value)

                loss_test_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_test)
                self.test_log.append(loss_test_value)

                pbar.set_postfix({'Max loss': loss_value.max(), 'Max test loss': loss_test_value.max()})

                self.params1_best, self.params2_best, self.Q10_best = v_model_selection(update, self.params1, self.params2, self.Q10, self.params1_best, self.params2_best, self.Q10_best)
                
    # Evaluates predictions at test points
    def posterior(self, x1, x2, x3, x4):
        normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
        denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

        # This is necessary so that for each network we get slightly different inputs given their normalizing factors
        x1 = jnp.tile(x1[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x2 = jnp.tile(x2[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x3 = jnp.tile(x3[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x4 = jnp.tile(x4[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        
        inputs1 = normalize(x1, self.mu_X1, self.sigma_X1)
        inputs2 = normalize(x2, self.mu_X2, self.sigma_X2)
        inputs3 = x3
        inputs4 = x4
        
        # For that reason we also vectorize over the ensemble axis for all
        samples, samples_GPP, samples_RECO, _ = vmap(self.net_forward_test, (0, 0, 0, 0, 0, 0, 0))(self.params1_best, self.params2_best, self.Q10_best, inputs1, inputs2, inputs3, inputs4)
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_GPP = denormalize(samples_GPP, self.mu_y, self.sigma_y)
        samples_RECO = denormalize(samples_RECO, self.mu_y, self.sigma_y)
        
        return samples, samples_GPP, samples_RECO, None
    

class Nightconstrained_Q10:
    def __init__(self, nn_1, nn_2, ensemble_size, p=0.01, weight_decay=0, rng_key=random.PRNGKey(0), Q10_mean_guess=1.5):
        self.init1, self.apply1, self.apply_eval1 = MLPDropout(nn_1)
        self.init2, self.apply2, self.apply_eval2 = MLPDropout(nn_2)
        self.Q10_mean_guess = Q10_mean_guess
        self.Q10_std_guess = 0.1
        self.ensemble_size = ensemble_size
        self.p = p
        
        # Random keys
        rng_key1, rng_key2, rng_key3, rng_key = random.split(rng_key, 4)
        
        keys_1 = random.split(rng_key1, ensemble_size)
        keys_2 = random.split(rng_key2, ensemble_size)
        
        # Initialize
        self.params1 = vmap(self.init1)(keys_1)
        self.params2 = vmap(self.init2)(keys_2)
        self.Q10 = self.Q10_mean_guess + self.Q10_std_guess * random.normal(rng_key3, (self.ensemble_size,1))
        self.Q10_init = self.Q10

        
        self.schedule1 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer1 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule1,
                                        weight_decay=weight_decay,
                                        ),
                            )
        self.opt_state1 = vmap(self.optimizer1.init)(self.params1)


        self.schedule2 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer2 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule2,
                                        weight_decay=0)
                            )
        self.opt_state2 = vmap(self.optimizer2.init)(self.params2)


        self.schedule_Q10 = optax.exponential_decay(init_value=0.1, transition_steps=500, decay_rate=0.95)

        self.optimizer_Q10 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule_Q10,
                                        weight_decay=0)
                            )

        self.opt_state_Q10 = vmap(self.optimizer_Q10.init)(self.Q10)
        #self.key_opt_state_Q10 = vmap(self.optimizer_Q10)(keys_4)


        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.Q10_log = []
        self.loss_test_log = [jnp.array(self.ensemble_size*[jnp.inf])]
        
        
    # Define the forward pass
    def net_forward(self, params1, params2, Q10, inputs1, inputs2, inputs3, inputs4, p, rng_key):
        rng_key1, rng_key2 = random.split(rng_key)
        
        GPP = jax.nn.softplus(self.apply1(params1, inputs1, p, rng_key1)) * jnp.abs(inputs3-1)
        Rb = jax.nn.softplus(self.apply2(params2, inputs2, p, rng_key2))
        RECO = Rb * Q10 ** (0.1 * (inputs4 - 15))

        Y_pred = -GPP + RECO
        return Y_pred

    def net_forward_test(self, params1, params2, Q10, inputs1, inputs2, inputs3, inputs4):
        
        GPP = jax.nn.softplus(self.apply_eval1(params1, inputs1)) * jnp.abs(inputs3-1)
        Rb = jax.nn.softplus(self.apply_eval2(params2, inputs2))
        RECO = Rb * Q10 ** (0.1 * (inputs4 - 15))

        Y_pred = -GPP + RECO
        return Y_pred, GPP, RECO, None


    # loss functions
    def loss(self, params1, params2, Q10, batch, p, rng_key):
        inputs1, inputs2, inputs3, inputs4, targets = batch
        
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, None, 0, 0, 0, 0, None, 0))(params1, params2, Q10,
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4,
                                                                            p, rng_key)
        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss

    def loss_test(self, params1, params2, Q10, batch):
        inputs1, inputs2, inputs3, inputs4, targets = batch

        # Compute forward pass
        outputs, _, _, _ = vmap(self.net_forward_test, (None, None, None, 0, 0, 0, 0))(params1, params2, Q10,
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4)

        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss


    # monitor loss functions
    def monitor_loss(self, params1, params2, Q10, batch, p, rng_key):
        loss_value = self.loss(params1, params2, Q10, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params1, params2, Q10, batch):
        loss_value = self.loss_test(params1, params2, Q10, batch)
        return loss_value

    # Define the update step
    def step(self, i, params1, params2, Q10, opt_state1, opt_state2, opt_state_Q10, batch, p, rng_key):
        grads1 = jax.grad(self.loss, argnums=0)(params1, params2, Q10, batch, p, rng_key)
        grads2 = jax.grad(self.loss, argnums=1)(params1, params2, Q10, batch, p, rng_key)
        grads_Q10 = jax.grad(self.loss, argnums=2)(params1, params2, Q10, batch, p, rng_key)
        updates1, opt_state1 = self.optimizer1.update(grads1, opt_state1, params1) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates2, opt_state2 = self.optimizer2.update(grads2, opt_state2, params2) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates_Q10, opt_state_Q10 = self.optimizer_Q10.update(grads_Q10, opt_state_Q10, Q10)
        
        params1 = optax.apply_updates(params1, updates1)
        params2 = optax.apply_updates(params2, updates2)
        Q10 = optax.apply_updates(Q10, updates_Q10)

        return params1, params2, Q10, opt_state1, opt_state2, opt_state_Q10
    
    def update_weights(self, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return params1, params2, Q10
        
    def keep_weights(self, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return params1_best, params2_best, Q10_best
    
    def model_selection(self, update, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return jax.lax.cond(update, self.update_weights, self.keep_weights, params1, params2, Q10, params1_best, params2_best, Q10_best)
    
    def batch_normalize(self, data_val, norm_const):
        X1, X2, X3, X4, y = data_val

        (mu_X1, sigma_X1), (mu_X2, sigma_X2), (mu_X3, sigma_X3), (mu_X4, sigma_X4), (mu_y, sigma_y) = norm_const
        X1 = (X1 - mu_X1)/sigma_X1
        X2 = (X2 - mu_X2)/sigma_X2
        #X3 = (X3 - mu_X3)/sigma_X3
        #X4 = (X4 - mu_X4)/sigma_X4
        y = (y - mu_y)/sigma_y
        
        return [X1, X2, X3, X4, y]
        
    # Optimize parameters in a loop
    def fit(self, dataset, data_train, data_val, data_test, rng_key, nIter = 1000):
        self.params1_best = self.params1
        self.params2_best = self.params2
        self.Q10_best = self.Q10

        data = iter(dataset)
        (self.mu_X1, self.sigma_X1), \
        (self.mu_X2, self.sigma_X2), \
        (self.mu_X3, self.sigma_X3), \
        (self.mu_X4, self.sigma_X4), \
        (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)

        # Vectorize along 0 axis which is ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0, 0, 0, None, 0)))   # vmap(step, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, 0, None, 0))) # vmap(monitor_loss, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes = (0, 0, 0, 0))) # vmap(monitor_loss_test, ensemble) -> loss_test -> vmap(net_forward_test, batch) (makes sense)
        
        v_model_selection = jit(vmap(self.model_selection, in_axes = (0, 0, 0, 0, 0, 0, 0))) # Completely ensemblewise, jitable?
        v_batch_normalize = vmap(self.batch_normalize, in_axes = (None, 0)) # Maybe dont jit because we apply it once

        data_train = v_batch_normalize(data_train, dataset.norm_const)
        data_val = v_batch_normalize(data_val, dataset.norm_const)
        data_test = v_batch_normalize(data_test, dataset.norm_const)

        # Main training loop
        for it in pbar:
            batch = next(data)
            rng_key, rng_key_ens_batch = random.split(rng_key, 2)
            rng_key_ens_batch = random.split(rng_key_ens_batch, batch[-1].shape[0]*batch[-1].shape[1]).reshape(batch[-1].shape[0], batch[-1].shape[1], 2)
            self.params1, self.params2, self.Q10, self.opt_state1, self.opt_state2, self.opt_state_Q10 = v_step(it, self.params1, self.params2, self.Q10,
                                                                        self.opt_state1, self.opt_state2, self.opt_state_Q10,
                                                                        batch, self.p, jnp.array(rng_key_ens_batch))
            
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10[:,0])
                loss_value = v_monitor_loss(self.params1, self.params2, self.Q10, batch, self.p, jnp.array(rng_key_ens_batch))
                self.loss_log.append(loss_value)
                
                loss_test_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                # Compute Values for logging
                loss_train_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_train)
                self.train_log.append(loss_train_value)

                loss_val_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_val)
                self.val_log.append(loss_val_value)

                loss_test_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_test)
                self.test_log.append(loss_test_value)

                pbar.set_postfix({'Max loss': loss_value.max(), 'Max test loss': loss_test_value.max()})

                self.params1_best, self.params2_best, self.Q10_best = v_model_selection(update, self.params1, self.params2, self.Q10, self.params1_best, self.params2_best, self.Q10_best)
                
    # Evaluates predictions at test points
    def posterior(self, x1, x2, x3, x4):
        normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
        denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

        # This is necessary so that for each network we get slightly different inputs given their normalizing factors
        x1 = jnp.tile(x1[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x2 = jnp.tile(x2[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x3 = jnp.tile(x3[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x4 = jnp.tile(x4[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        
        inputs1 = normalize(x1, self.mu_X1, self.sigma_X1)
        inputs2 = normalize(x2, self.mu_X2, self.sigma_X2)
        inputs3 = x3
        inputs4 = x4
        
        # For that reason we also vectorize over the ensemble axis for all
        samples, samples_GPP, samples_RECO, _ = vmap(self.net_forward_test, (0, 0, 0, 0, 0, 0, 0))(self.params1_best, self.params2_best, self.Q10_best, inputs1, inputs2, inputs3, inputs4)
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_GPP = denormalize(samples_GPP, self.mu_y, self.sigma_y)
        samples_RECO = denormalize(samples_RECO, self.mu_y, self.sigma_y)
        
        return samples, samples_GPP, samples_RECO, None


class LUE_Q10:
    def __init__(self, nn_1, nn_2, ensemble_size, p=0.01, weight_decay=0, rng_key=random.PRNGKey(0), Q10_mean_guess=1.5):
        self.init1, self.apply1, self.apply_eval1 = MLPDropout(nn_1)
        self.init2, self.apply2, self.apply_eval2 = MLPDropout(nn_2)
        self.Q10_mean_guess = Q10_mean_guess
        self.Q10_std_guess = 0.1
        self.ensemble_size = ensemble_size
        self.p = p
        
        # Random keys
        rng_key1, rng_key2, rng_key3, rng_key = random.split(rng_key, 4)
        
        keys_1 = random.split(rng_key1, ensemble_size)
        keys_2 = random.split(rng_key2, ensemble_size)
        
        # Initialize
        self.params1 = vmap(self.init1)(keys_1)
        self.params2 = vmap(self.init2)(keys_2)
        self.Q10 = self.Q10_mean_guess + self.Q10_std_guess * random.normal(rng_key3, (self.ensemble_size,1))
        self.Q10_init = self.Q10

        
        self.schedule1 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer1 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule1,
                                        weight_decay=weight_decay,
                                        ),
                            )
        self.opt_state1 = vmap(self.optimizer1.init)(self.params1)


        self.schedule2 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)
        self.optimizer2 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule2,
                                        weight_decay=0)
                            )
        self.opt_state2 = vmap(self.optimizer2.init)(self.params2)


        self.schedule_Q10 = optax.exponential_decay(init_value=0.01, transition_steps=500, decay_rate=0.95)

        self.optimizer_Q10 = optax.chain(
                            #optax.clip(1.0),
                            optax.adamw(learning_rate=self.schedule_Q10,
                                        weight_decay=0)
                            )

        self.opt_state_Q10 = vmap(self.optimizer_Q10.init)(self.Q10)
        #self.key_opt_state_Q10 = vmap(self.optimizer_Q10)(keys_4)


        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.Q10_log = []
        self.lr_log = []
        self.best_i = []
        self.loss_test_log = [jnp.array(self.ensemble_size*[jnp.inf])]
        
        
    # Define the forward pass
    def net_forward(self, params1, params2, Q10, inputs1, inputs2, inputs3, inputs4, p, rng_key):
        rng_key1, rng_key2 = random.split(rng_key)
        
        GPP = jax.nn.softplus(self.apply1(params1, inputs1, p, rng_key1)) * inputs3
        Rb = jax.nn.softplus(self.apply2(params2, inputs2, p, rng_key2))
        RECO = Rb * Q10 ** (0.1 * (inputs4 - 15))

        Y_pred = -GPP + RECO
        return Y_pred

    def net_forward_test(self, params1, params2, Q10, inputs1, inputs2, inputs3, inputs4):
        
        LUE = jax.nn.softplus(self.apply_eval1(params1, inputs1))
        GPP = LUE*inputs3
        Rb = jax.nn.softplus(self.apply_eval2(params2, inputs2))
        RECO = Rb * Q10 ** (0.1 * (inputs4 - 15))

        Y_pred = -GPP + RECO
        return Y_pred, GPP, RECO, LUE


    # loss functions
    def loss(self, params1, params2, Q10, batch, p, rng_key):
        inputs1, inputs2, inputs3, inputs4, targets = batch
        
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, None, 0, 0, 0, 0, None, 0))(params1, params2, Q10,
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4,
                                                                            p, rng_key)
        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss

    def loss_test(self, params1, params2, Q10, batch):
        inputs1, inputs2, inputs3, inputs4, targets = batch

        # Compute forward pass
        outputs, _, _, _ = vmap(self.net_forward_test, (None, None, None, 0, 0, 0, 0))(params1, params2, Q10,
                                                                            inputs1, inputs2,
                                                                            inputs3, inputs4)

        # Compute loss
        loss = jnp.mean((targets - outputs)**2)
        return loss


    # monitor loss functions
    def monitor_loss(self, params1, params2, Q10, batch, p, rng_key):
        loss_value = self.loss(params1, params2, Q10, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params1, params2, Q10, batch):
        loss_value = self.loss_test(params1, params2, Q10, batch)
        return loss_value

    # Define the update step
    def step(self, i, params1, params2, Q10, opt_state1, opt_state2, opt_state_Q10, batch, p, rng_key):
        grads1 = jax.grad(self.loss, argnums=0)(params1, params2, Q10, batch, p, rng_key)
        grads2 = jax.grad(self.loss, argnums=1)(params1, params2, Q10, batch, p, rng_key)
        grads_Q10 = jax.grad(self.loss, argnums=2)(params1, params2, Q10, batch, p, rng_key)
        updates1, opt_state1 = self.optimizer1.update(grads1, opt_state1, params1) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates2, opt_state2 = self.optimizer2.update(grads2, opt_state2, params2) # Note that opt_state keeps track of the state of the optimizer, optimizer seem to be static and don't change
        updates_Q10, opt_state_Q10 = self.optimizer_Q10.update(grads_Q10, opt_state_Q10, Q10)
        
        params1 = optax.apply_updates(params1, updates1)
        params2 = optax.apply_updates(params2, updates2)
        Q10 = optax.apply_updates(Q10, updates_Q10)

        return params1, params2, Q10, opt_state1, opt_state2, opt_state_Q10
    
    def update_weights(self, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return params1, params2, Q10
        
    def keep_weights(self, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return params1_best, params2_best, Q10_best
    
    def model_selection(self, update, params1, params2, Q10, params1_best, params2_best, Q10_best):
        return jax.lax.cond(update, self.update_weights, self.keep_weights, params1, params2, Q10, params1_best, params2_best, Q10_best)
    
    def batch_normalize(self, data_val, norm_const):
        X1, X2, X3, X4, y = data_val

        (mu_X1, sigma_X1), (mu_X2, sigma_X2), (mu_X3, sigma_X3), (mu_X4, sigma_X4), (mu_y, sigma_y) = norm_const
        X1 = (X1 - mu_X1)/sigma_X1
        X2 = (X2 - mu_X2)/sigma_X2
        X3 = (X3 - mu_X3)/sigma_X3
        #X4 = (X4 - mu_X4)/sigma_X4
        y = (y - mu_y)/sigma_y
        
        return [X1, X2, X3, X4, y]
        
    # Optimize parameters in a loop
    def fit(self, dataset, data_train, data_val, data_test, rng_key, nIter = 1000):
        self.params1_best = self.params1
        self.params2_best = self.params2
        self.Q10_best = self.Q10

        data = iter(dataset)
        (self.mu_X1, self.sigma_X1), \
        (self.mu_X2, self.sigma_X2), \
        (self.mu_X3, self.sigma_X3), \
        (self.mu_X4, self.sigma_X4), \
        (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)

        # Vectorize along 0 axis which is ensemble
        v_step = jit(vmap(self.step, in_axes = (None, 0, 0, 0, 0, 0, 0, 0, None, 0)))   # vmap(step, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes = (0, 0, 0, 0, None, 0))) # vmap(monitor_loss, ensemble) -> loss -> vmap(net_forward, batch) (makes sense)
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes = (0, 0, 0, 0))) # vmap(monitor_loss_test, ensemble) -> loss_test -> vmap(net_forward_test, batch) (makes sense)
        
        v_model_selection = jit(vmap(self.model_selection, in_axes = (0, 0, 0, 0, 0, 0, 0))) # Completely ensemblewise, jitable?
        v_batch_normalize = vmap(self.batch_normalize, in_axes = (None, 0)) # Maybe dont jit because we apply it once

        data_train = v_batch_normalize(data_train, dataset.norm_const)
        data_val = v_batch_normalize(data_val, dataset.norm_const)
        data_test = v_batch_normalize(data_test, dataset.norm_const)

        # Main training loop
        for it in pbar:
            batch = next(data)
            rng_key, rng_key_ens_batch = random.split(rng_key, 2)
            rng_key_ens_batch = random.split(rng_key_ens_batch, batch[-1].shape[0]*batch[-1].shape[1]).reshape(batch[-1].shape[0], batch[-1].shape[1], 2)
            self.params1, self.params2, self.Q10, self.opt_state1, self.opt_state2, self.opt_state_Q10 = v_step(it, self.params1, self.params2, self.Q10,
                                                                        self.opt_state1, self.opt_state2, self.opt_state_Q10,
                                                                        batch, self.p, jnp.array(rng_key_ens_batch))
            
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10[:,0])
                loss_value = v_monitor_loss(self.params1, self.params2, self.Q10, batch, self.p, jnp.array(rng_key_ens_batch))
                self.loss_log.append(loss_value)
                
                loss_test_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                # Compute Values for logging
                loss_train_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_train)
                self.train_log.append(loss_train_value)

                loss_val_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_val)
                self.val_log.append(loss_val_value)

                loss_test_value = v_monitor_loss_test(self.params1, self.params2, self.Q10, data_test)
                self.test_log.append(loss_test_value)

                pbar.set_postfix({'Max loss': loss_value.max(), 'Max test loss': loss_test_value.max()})

                self.params1_best, self.params2_best, self.Q10_best = v_model_selection(update, self.params1, self.params2, self.Q10, self.params1_best, self.params2_best, self.Q10_best)
                
    # Evaluates predictions at test points
    def posterior(self, x1, x2, x3, x4):
        normalize = vmap(lambda x, mu, std: (x-mu)/std, in_axes=(0,0,0))
        denormalize = vmap(lambda x, mu, std: x*std + mu, in_axes=(0,0,0))

        # This is necessary so that for each network we get slightly different inputs given their normalizing factors
        x1 = jnp.tile(x1[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x2 = jnp.tile(x2[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x3 = jnp.tile(x3[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        x4 = jnp.tile(x4[jnp.newaxis,:,:], (self.ensemble_size, 1, 1))
        
        inputs1 = normalize(x1, self.mu_X1, self.sigma_X1)
        inputs2 = normalize(x2, self.mu_X2, self.sigma_X2)
        inputs3 = normalize(x3, self.mu_X3, self.sigma_X3)
        inputs4 = x4
        
        # For that reason we also vectorize over the ensemble axis for all
        samples, samples_GPP, samples_RECO, samples_LUE = vmap(self.net_forward_test, (0, 0, 0, 0, 0, 0, 0))(self.params1_best, self.params2_best, self.Q10_best, inputs1, inputs2, inputs3, inputs4)
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_GPP = denormalize(samples_GPP, self.mu_y, self.sigma_y)
        samples_RECO = denormalize(samples_RECO, self.mu_y, self.sigma_y)
        
        return samples, samples_GPP, samples_RECO, samples_LUE
