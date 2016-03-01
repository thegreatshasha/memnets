import tensorflow as tf
import numpy as np

def network():
    # porting the dataflow right now. we are using just a single hop and evaluating it on babl toy tasks

    # dimensions
    D = 3 # embedding space dimension
    N = 2 # no of inputs/memories
    V = 4 # vocabulary size
    L = 4

    # data placeholder
    #x = tf.placeholder(tf.float32, shape=(N, V, L))
    #q = tf.placeholder(tf.float32, shape=(V, L))
    with tf.name_scope("input") as scope:
        x = tf.constant([[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]],[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]]], dtype=tf.float32, name='x')
        q = tf.constant([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]], dtype=tf.float32, name='q')

    # Define the embedding matrices as variable
    with tf.name_scope("embedding_matrices") as scope:
        A = tf.Variable(tf.ones([D, V]), name='A') # input embedding matrix, D*V
        B = tf.Variable(tf.ones([D, V]), name='B') # questions embedding matrix, D*V
        C = tf.Variable(tf.ones([D, V]), name='C') # input embedding matrix, D*V

        W = tf.Variable(tf.ones([D, V]), name='W') # prediction embedding matrix, D*V

    # convert embedding matrices to 3d
    with tf.name_scope("embed_matrix_to_3d") as scope:
        A_3d = tf.reshape(tf.tile(tf.reshape(A, [D*V]), [N]), shape=[N,D,V], name='A_3d')
        C_3d = tf.reshape(tf.tile(tf.reshape(C, [D*V]), [N]), shape=[N,D,V], name='C_3d')

    # embedding everything
    with tf.name_scope("one_hot_to_embedded") as scope:
        m = tf.batch_matmul(A_3d, x)
        c = tf.batch_matmul(C_3d, x)
        u = tf.matmul(W,q)

    # positional and temporal encoding
    # skipping this for now

    # sum the elmements along horizontal axis for getting bag of words representation
    cnet = tf.transpose(tf.reduce_sum(c, 2)) # D*N
    mnet = tf.transpose(tf.reduce_sum(m, 2)) # D*N
    unet = tf.reduce_sum(u) # D*1
    #
    # # get softmax probabilities for each memory with input
    # scores = np.dot(tf.transpose(mnet), unet)# get match score of u with each of the memories, N*1
    # p = tf.nn.softmax(scores) # N*1
    #
    # # get prediction
    # o = tf.matmul(cnet, p) # D*1
    # pred = tf.matmul(tf.transpose(W), (o + unet)) # V*1

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        zeton = sess.run(init)
        #import pdb; pdb.set_trace()
        print "y"
        tf.train.SummaryWriter("data/", sess.graph.as_graph_def(add_shapes=True))
        #tf.train.write_graph(sess.graph_def, './', 'graph.pbtxt')

if __name__ == "__main__":
    network()
