import os
import tempfile

import pandas as pd
import streamlit as st
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
import time
import difflib
import keras.backend as K
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from deap import base, creator, tools, algorithms
from deap import tools
import random
import matplotlib.pyplot as plt
import seaborn as sns




