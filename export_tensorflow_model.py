import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d",  "--dir", type=str, help="base directory", required=True)
parser.add_argument("-i",  "--input", type=str, help="inputfile", required=True)
parser.add_argument("-o",  "--output", type=str, help="name", required=True)
parser.add_argument("-m",  "--tf_metadata", type=str, help="tf metadata file", default="tf_metadata.txt")
parser.add_argument("-is", "--input_scaler", type=str, help="input path to scaler file")
parser.add_argument("-os", "--output_scaler", type=str, help="name for output scaler file")
args = parser.parse_args()

from keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

from keras.models import load_model
model = load_model(os.path.join(args.dir, args.input))
print("output tensors: ", model.outputs[0].name)
print("input tensors: ", model.inputs[0].name)

from keras import backend as K
import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# Save to ./model/tf_model.pb
tf.train.write_graph(frozen_graph, args.dir, args.output, as_text=False)

## save tensorflow metadata
with open(os.path.join(args.dir, args.tf_metadata), "w") as f:
    f.write(str(model.inputs[0].name) + " " + str(model.outputs[0].name) + "\n")


## Export 
## * tf tensor input name and output name
## * scaler mean_ and scale_ (where scale_ = np.sqrt(var_)) for each variable
##
import os
import pickle

import yaml
yaml_vars = yaml.safe_load(open(os.path.join(args.dir, "variables.yml"), "r"))
print(type(yaml_vars), yaml_vars)

def export_scaler():
    if args.input_scaler:
        if not args.output_scaler:
            print("no output_scaler parameter provided")
            return
        scaler = pickle.load(open(os.path.join(args.dir, args.input_scaler), 'rb'))
        print(scaler.mean_)
        print(scaler.scale_)
        with open(os.path.join(args.dir, args.output_scaler), "w") as f:
            for var, mean, scale in zip(yaml_vars, scaler.mean_, scaler.scale_):
                f.write(var + " " + str(mean) + " " + str(scale) + "\n")


export_scaler()

print("done")