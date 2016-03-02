import tensorflow as tf
import numpy as np
from tasks import inputt,question,n,vocab, answer

#print x,q,n,vocab, answer
# Not really sure what's the best way to do this
D = 4
N = inputt.shape[0] # no of inputs/memories
V = len(vocab) # vocabulary size
L = inputt.shape[2] # max sentence length

def network():
    # porting the dataflow right now. we are using just a single hop and evaluating it on babl toy tasks

    # data placeholder
    with tf.name_scope("input") as scope:
        x = tf.placeholder(tf.float32, name='x')
        q = tf.placeholder(tf.float32, name='q')
        y = tf.placeholder(tf.float32, name='y')
        #x = tf.constant([[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]],[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]]], dtype=tf.float32, name='x')
        #q = tf.constant([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]], dtype=tf.float32, name='q')

    # Define dimensions

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
    unet = tf.reduce_sum(u, 1, keep_dims=True) # D*1
    #
    # # get softmax probabilities for each memory with input
    scores = tf.matmul(tf.transpose(mnet), unet)# get match score of u with each of the memories, N*1
    p = tf.nn.softmax(scores) # N*1
    #
    # # get prediction
    o = tf.matmul(cnet, p) # D*1
    pred = tf.nn.softmax(tf.matmul(tf.transpose(W), (o + unet))) # V*1

    # get cross entropy erro
    cross_entropy = -tf.reduce_sum(y * tf.log(pred))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(cross_entropy)

    init = tf.initialize_all_variables()

    # Run graph session
    sess = tf.Session()
    sess.run(init)

    # Prepare the event logging
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('data',
                                        graph_def=sess.graph_def)


    # Fit the training data
    for step in range(10):
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        summary_str = sess.run(train, feed_dict={x: inputt, y: answer, q: question})
        summary_writer.add_summary(summary_str, step)
    #tf.train.SummaryWriter("data/", sess.graph.as_graph_def(add_shapes=True))

if __name__ == "__main__":
    network()
