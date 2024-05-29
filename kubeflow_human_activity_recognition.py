# Import the modules you will use
import kfp

# For creating the pipeline
from kfp.v2 import dsl

# For building components
from kfp.v2.dsl import component

# Type annotations for the component artifacts
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics
)

from kfp import compiler

#data ingestion and formatting

@component(
    packages_to_install=["pandas", "openpyxl", "scikit-learn", "numpy"],
    base_image="python:3.8",
    output_component_file="clean_data_component.yaml"
)
def clean_data(path: str, output_csv: Output[Dataset]):
    
    def convert_to_float(x):
        try:
            return np.float(x)
        except:
            return 0.0
    
    column_names = ['user-id', 'activity', 'timestamp', 'x-acc', 'y-acc', 'z-acc']
    df = pd.read_csv(path, header=None, names=column_names)
    df['z-acc'].replace(regex=True, inplace=True, to_replace=r';', value=r'')
    df['z-acc'] = df['z-acc'].apply(convert_to_float)
    
    df.dropna(axis=0, how='any', inplace=True)
    df.to_csv(output_csv.path, index=False)

    
#spliting the data into test and train

@component(
    packages_to_install=["pandas", "scikit-learn"],
    output_component_file="split_data_component.yaml"
)
def split_data(input_csv: Input[Dataset], train_csv: Output[Dataset], test_csv: Output[Dataset]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(input_csv.path)
    
    total_rows = len(df)
    split_index = int(0.7 * total_rows)
    
    train_df = df.iloc[:split_index]   
    test_df = df.iloc[split_index:] 
    
    train_df.to_csv(train_csv.path, index=False)
    test_df.to_csv(test_csv.path, index=False)
    
#converting the format of data to time series for prediction of activity 
#based on the mode of activity in respective window

@component(
    packages_to_install=["pandas", "numpy", "scipy", "scikit-learn"],
    output_component_file="transform_data_component.yaml"
)
def preprocess_data(
    input_train_csv: Input[Dataset], 
    input_test_csv: Input[Dataset], 
    output_train_x: Output[Artifact], 
    output_test_x: Output[Artifact],
    output_train_y: Output[Artifact], 
    output_test_y: Output[Artifact]
):
    from scipy import stats
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import numpy as np
    import pandas as pd
    import pickle

    LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
    TIME_PERIODS = 80
    STEP_DISTANCE = 40
    LABEL = 'activity'
    N_FEATURES = 3
    acc_cols = ['x-acc', 'y-acc', 'z-acc']

    train = pd.read_csv(input_train_csv.path)
    test = pd.read_csv(input_test_csv.path)

    for col in acc_cols:
        scaler = MinMaxScaler()
        train[col] = scaler.fit_transform(train[[col]])
        test[col] = scaler.transform(test[[col]])

    label_encoder = LabelEncoder()
    train.loc[:, LABEL] = label_encoder.fit_transform(train[LABEL].values.ravel())
    test.loc[:, LABEL] = label_encoder.transform(test[LABEL].values.ravel())

    def create_segments_and_labels(df, time_period, step_distance, label_name):
        segments = []
        labels = []
        for i in range(0, len(df) - time_period, step_distance):
            xs = df['x-acc'].values[i: i + time_period]
            ys = df['y-acc'].values[i: i + time_period]
            zs = df['z-acc'].values[i: i + time_period]

            label = stats.mode(df[label_name][i: i + time_period])[0][0]
            segments.append([xs, ys, zs])
            labels.append(label)
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_period, N_FEATURES)
        labels = np.asarray(labels)
        return reshaped_segments, labels

    X_train, Y_train = create_segments_and_labels(train, TIME_PERIODS, STEP_DISTANCE, LABEL)
    X_test, Y_test = create_segments_and_labels(test, TIME_PERIODS, STEP_DISTANCE, LABEL)

    with open(output_train_x.path, 'wb') as f:
        pickle.dump(X_train, f)
    with open(output_test_x.path, 'wb') as f:
        pickle.dump(X_test, f)

    with open(output_train_y.path, 'wb') as f:
        pickle.dump(Y_train, f)
    with open(output_test_y.path, 'wb') as f:
        pickle.dump(Y_test, f)

#tranning a conv-1d model on the train_data 

@component(
    packages_to_install=["tensorflow", "pandas"],
    output_component_file="train_model_component.yaml"
)
def train_model(
    input_train_x: Input[Artifact], 
    input_train_y: Input[Artifact], 
    output_model: Output[Model], 
    output_history: Output[Artifact]
):
    import tensorflow as tf
    from tensorflow.keras import models, layers
    from tensorflow.keras.utils import to_categorical
    import pickle

    with open(input_train_x.path, "rb") as file:
        train_X = pickle.load(file)

    with open(input_train_y.path, "rb") as file:
        train_Y = pickle.load(file)

    Y_one_hot = to_categorical(train_Y)

    def model_builder(train_X):
        model = models.Sequential()
        model.add(layers.Conv1D(160, 12, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))
        model.add(layers.Conv1D(128, 10, activation='relu'))
        model.add(layers.Conv1D(96, 8, activation='relu'))
        model.add(layers.Conv1D(64, 6, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(6, activation='softmax'))
        print(model.summary())
        return model

    model = model_builder(train_X)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_X, Y_one_hot, epochs=25, batch_size=1024)
    model.save(output_model.path)

    with open(output_history.path, "wb") as file:
        pickle.dump(history.history, file)

#evaluation the model on test data and return metrics as logs of pipeline output

@component(
    packages_to_install=["tensorflow", "pandas"],
    output_component_file="eval_model_component.yaml"
)
def eval_model(
    input_model: Input[Model], 
    input_history: Input[Artifact], 
    input_test_x: Input[Artifact], 
    input_test_y: Input[Artifact], 
    MLPipeline_Metrics: Output[Metrics]
):
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    import pickle

    model = tf.keras.models.load_model(input_model.path)

    with open(input_test_x.path, "rb") as file:
        test_X = pickle.load(file)

    with open(input_test_y.path, "rb") as file:
        test_Y = pickle.load(file)

    Y_one_hot = to_categorical(test_Y)

    loss_value, accuracy = model.evaluate(test_X, Y_one_hot)
    output_string = f"Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}"

    MLPipeline_Metrics.log_metric("categorical_crossentropy_loss", loss_value)
    MLPipeline_Metrics.log_metric("accuracy", accuracy)

#completely defining the pipeline dag /structure

@dsl.pipeline(
    name="Human Activity Recognition Pipeline",
)
def kfp_pipeline(data_path: str):
    
    clean_data_task = clean_data(path=data_path)
    
    split_data_task = split_data(input_csv=clean_data_task.outputs['output_csv'])
    
    preprocess_data_task = preprocess_data(
        input_train_csv=split_data_task.outputs['train_csv'],
        input_test_csv=split_data_task.outputs['test_csv']
    )
    
    train_model_task = train_model(
        input_train_x=preprocess_data_task.outputs["output_train_x"],
        input_train_y=preprocess_data_task.outputs["output_train_y"]
    )
    
    eval_model_task = eval_model(
        input_model=train_model_task.outputs["output_model"],
        input_history=train_model_task.outputs["output_history"],
        input_test_x=preprocess_data_task.outputs["output_test_x"],
        input_test_y=preprocess_data_task.outputs["output_test_y"]
    )


#compiling the pipeline into yaml to run on kubeflow
if __name__ == "__main__":
    compiler.Compiler().compile(kfp_pipeline, 'pipeline.yaml')
