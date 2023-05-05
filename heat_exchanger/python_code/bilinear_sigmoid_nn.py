import pandas as pd
import pickle
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from omlt import OmltBlock, OffsetScaling
from omlt.io import load_keras_sequential
from omlt.neuralnet import FullSpaceSmoothNNFormulation
from omlt.neuralnet import ReducedSpaceSmoothNNFormulation

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

# we train 'n_instance' neural networks with randomly selected data
np.random.seed(12345)

n_instance = 5
T5 = [i for i in range(573,593,1)]

opt_duty = dict()
opt_duty['global_reduced'] = np.empty([len(T5), n_instance], dtype = 'float')
opt_duty['local_reduced'] = np.empty([len(T5), n_instance], dtype = 'float')
opt_duty['global_fullspace'] = np.empty([len(T5), n_instance], dtype = 'float')
opt_duty['local_fullspace'] = np.empty([len(T5), n_instance], dtype = 'float')

solve_time = dict()
solve_time['global_reduced'] = np.empty([len(T5), n_instance], dtype = 'float')
solve_time['local_reduced'] = np.empty([len(T5), n_instance], dtype = 'float')
solve_time['global_fullspace'] = np.empty([len(T5), n_instance], dtype = 'float')
solve_time['local_fullspace'] = np.empty([len(T5), n_instance], dtype = 'float')

def optimize_he(T5, method, formulation):
    # first, create the Pyomo model
    m = pyo.ConcreteModel()
    # create the OmltBlock to hold the neural network model
    m.ratelaw = OmltBlock()
    # create a network definition from the Keras model
    unscaled_net = load_keras_sequential(nn, scaling_object=scaler, scaled_input_bounds=scaled_input_bounds)

    # create the variables and constraints for the neural network in Pyomo
    if formulation == "reduced":
        m.ratelaw.build_formulation(ReducedSpaceSmoothNNFormulation(unscaled_net))
    else:
        m.ratelaw.build_formulation(FullSpaceSmoothNNFormulation(unscaled_net))

    Qc_idx = inputs.index('Qc')
    FH_idx = inputs.index('FH')
    bil_idx = outputs.index('QcFH')
  
    # Objective
    m.cost = pyo.Objective(expr = 0.01*m.ratelaw.inputs[Qc_idx] + 4*(m.ratelaw.inputs[FH_idx] - 1.7)**2, sense = pyo.minimize)

    # Constraints
    m.con1 = pyo.Constraint(expr = -10 - m.ratelaw.inputs[Qc_idx] + (T5 - 558)*m.ratelaw.inputs[FH_idx] + 0.5*m.ratelaw.outputs[bil_idx] >= 0)
    m.con2 = pyo.Constraint(expr = -10 - m.ratelaw.inputs[Qc_idx] + (T5 - 393)*m.ratelaw.inputs[FH_idx] >= 0)
    m.con3 = pyo.Constraint(expr = -250 - m.ratelaw.inputs[Qc_idx] + (T5 - 313)*m.ratelaw.inputs[FH_idx] >= 0)
    m.con4 = pyo.Constraint(expr = -250 - m.ratelaw.inputs[Qc_idx] + (T5 - 323)*m.ratelaw.inputs[FH_idx] <= 0)
    m.con5 = pyo.Constraint(expr = m.ratelaw.inputs[Qc_idx] >= 0)
    m.con6 = pyo.Constraint(expr = m.ratelaw.inputs[FH_idx] >= 0)

    # m.pprint()
    if method == "local":
        solver = pyo.SolverFactory('ipopt', executable='ipopt').solve(m, tee=False)
    else:
        solver = pyo.SolverFactory('baron', executable='~/BARON/baron').solve(m, tee=False)
    
    # m.load(solver) # Loading solution into solver object

    if (solver.solver.status == SolverStatus.ok) and (solver.solver.termination_condition == TerminationCondition.optimal):
        # Do nothing when the solution in optimal and feasible
        print('optimal')
    elif (solver.solver.termination_condition == TerminationCondition.infeasible):
        print('infeasible')
    else:
        print('model did not solve correctly.')
        print("Solver Status: ",  solver.solver.status)

    return pyo.value(m.ratelaw.inputs[Qc_idx]()), solver['Solver'][0]['Time']

sigmoid_nn_data = {'duty': opt_duty, 'time': solve_time, 'nn_predictions': []}

columns = ['Qc', 'FH', 'QcFH']
df = pd.read_csv('bilinear_data.csv', usecols=columns)
inputs = ['Qc', 'FH']
outputs = ['QcFH']
dfin = df[inputs]
dfout = df[outputs]

prediction_data = []

for instance in range(n_instance):
    df = pd.read_csv('bilinear_data.csv', usecols=columns)
    dfin = df[inputs]
    dfout = df[outputs]

    # create our Sequential model
    nn = Sequential(name='nn_sigmoid_2_10')
    nn.add(Dense(units=10, input_dim=len(inputs), activation='sigmoid'))
    nn.add(Dense(units=10, activation='sigmoid'))
    # nn.add(Dense(units=10, activation='sigmoid'))
    # nn.add(Dense(units=10, activation='sigmoid'))
    nn.add(Dense(units=len(outputs)))
    nn.compile(optimizer=Adam(), loss='mse')

    # train our model with random data
    n_train = 1000
    train_idx = np.random.choice(dfin.shape[0], n_train, replace=False)
    
    # scale the inputs and outputs
    x_offset, x_factor = dfin.iloc[train_idx].mean().to_dict(), dfin.iloc[train_idx].std().to_dict()
    y_offset, y_factor = dfout.iloc[train_idx].mean().to_dict(), dfout.iloc[train_idx].std().to_dict()
    dfin = (dfin - dfin.iloc[train_idx].mean()).divide(dfin.iloc[train_idx].std())
    dfout = (dfout - dfout.iloc[train_idx].mean()).divide(dfout.iloc[train_idx].std())

    scaled_lb = dfin.min()[inputs].values
    scaled_ub = dfin.max()[inputs].values

    global scaled_input_bounds
    scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}

    # Create OMLT object required to unscale the nn output later
    global scaler
    scaler = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

    x = dfin.iloc[train_idx].values
    y = dfout.iloc[train_idx].values

    n_test = 100
    test_idx = np.random.choice(dfin.shape[0], n_test, replace=False)
    x_test = dfin.iloc[test_idx].values
    y_test = dfout.iloc[test_idx].values

    history = nn.fit(x, y, epochs=1000, verbose = False)

    # Check prediction accuracy
    predictions = nn(x_test)
    y_test = np.reshape(y_test, n_test)
    predictions = np.reshape(predictions, n_test)

    prediction_data.append(np.transpose(np.array([y_test, predictions])))
    nn.save("nn_instance_" + str(instance))
    
    # solve the optimization problems
    for j in range(len(T5)):
        print("T5 = ", T5[j])
        print('1. local_reduced: ')
        sigmoid_nn_data['duty']['local_reduced'][j, instance], sigmoid_nn_data['time']['local_reduced'][j, instance] = \
        optimize_he(T5[j], "local", "reduced")
        print('2. global_reduced: ')
        sigmoid_nn_data['duty']['global_reduced'][j, instance], sigmoid_nn_data['time']['global_reduced'][j, instance] = \
        optimize_he(T5[j], "global", "reduced")
        print('3. local_fullspace: ')
        sigmoid_nn_data['duty']['local_fullspace'][j, instance], sigmoid_nn_data['time']['local_fullspace'][j, instance] = \
        optimize_he(T5[j], "local", "fullspace")
        print('4. global_fullspace: ')
        sigmoid_nn_data['duty']['global_fullspace'][j, instance], sigmoid_nn_data['time']['global_fullspace'][j, instance] = \
        optimize_he(T5[j], "global", "fullspace")

sigmoid_nn_data['nn_predictions'] = prediction_data
with open('bilinear_nn_sigmoid_data_rescaled.pickle', 'wb') as file:
    pickle.dump(sigmoid_nn_data, file)
file.close()