import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2

###
# This script removes the dropout layers that are used to train the model, but are not necessary to 
# perform inference. Removing the dropout layers and the input keep_prob variable reduces the model
# size and makes it easier to port from application to application.
###

# Main function: Searches through the network graph for layers used for dropout and removes them
def remove_dropout(args):
    input_model = os.path.join(os.getcwd(), args.model_directory, args.model_file)
    graph = tf.GraphDef()

    dropout_start = None
    dropout_end = None
    final_node = None
    first_instance = True
    keep_prob_index = None

    # Read the network graph
    with tf.gfile.Open(input_model, 'rb') as f:
        data = f.read()
        graph.ParseFromString(data)

    # Uncomment to print the network structure before dropout removal
    #for i, node in enumerate(graph.node):
    #    print('%d %s' % (i, node.name))
    #    for j, inputs in enumerate(node.input):
    #        print('--> %d, %s' % (j, inputs))

    # Iterate through the graph keeping track of which nodes to remove
    for i, node in enumerate(graph.node):
        if 'dropout' in node.name:
            if first_instance:
                dropout_start = i
                first_instance = False
            else:
                dropout_end = i
        if 'keep_prob' in node.name:
            keep_prob_index = i
        if args.output_node == node.name:
            final_node = i

    # Find the input into dropout
    dropout_input = graph.node[dropout_start].input[0]
    
    # Find the dropout output
    output_node = None
    output_node_input = None
    for i, node in enumerate(graph.node):
        for j, inputs in enumerate(node.input):
            if inputs == graph.node[dropout_end].name:
                output_node = i
                output_node_input = j
                
    # Link the dropout input with dropout output
    graph.node[output_node].input[output_node_input] = dropout_input
    # Remove dropout from graph
    final_graph = graph.node[:dropout_start] + graph.node[dropout_end + 1:final_node + 1]
    # Remove keep_prob from graph
    del final_graph[keep_prob_index] 

    # Uncomment to print the network structore after dropout removal
    #for i, node in enumerate(final_graph):
    #    print('%d %s' % (i, node.name))
    #    for j, inputs in enumerate(node.input):
    #        print('--> %d, %s' % (j, inputs))

    # Saves the new model
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(final_graph)
    with tf.gfile.GFile(args.output_model_file, 'wb') as f:
        f.write(output_graph.SerializeToString())
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", default="output", help="name of directory containing frozen model")
    parser.add_argument("--model_file", default="my_model.pb", help="name of frozen model file")
    parser.add_argument("--output_node", default="output", help="name of final output node in model")
    parser.add_argument("--output_model_file", default="nodropout.pb", help="name of output mode with dropout removed")
    args = parser.parse_args()
    remove_dropout(args)
