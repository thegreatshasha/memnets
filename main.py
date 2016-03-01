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
    x = tf.constant([[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]],[[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,0]]])


    # Define the embedding matrices as variable
    # A = tf.Variable(tf.ones([D, V])) # input embedding matrix, D*V
    # B = tf.Variable(tf.ones([D, V])) # questions embedding matrix, D*V
    # C = tf.Variable(tf.ones([D, V])) # input embedding matrix, D*V
    W = tf.Variable(tf.ones([D, V])) # prediction embedding matrix, D*V
    A2 = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) # input embedding matrix, D*V

    # question embedding, D*L
    a3 = tf.reshape(tf.tile(tf.reshape(A2, [D*V]), [N]), [N,D,V])
    #u = tf.matmul(W,q)
    #
    # # input embeddings, N*D*L
    m = tf.batch_matmul(a3, x)
    # c = tf.matmul(C, x).swapaxes(0, 1)
    #
    # # positional and temporal encoding
    # # skipping this for now
    #
    # # sum the elmements along horizontal axis for getting bag of words representation
    # cnet = c.sum(axis=2).transpose() # D*N
    # mnet = m.sum(axis=2).transpose() # D*N
    # unet = u.sum() # D*1
    #
    # # get softmax probabilities for each memory with input
    # scores = np.dot(tf.transpose(mnet), unet)# get match score of u with each of the memories, N*1
    # p = tf.nn.softmax(scores) # N*1
    #
    # # get prediction
    # o = tf.matmul(cnet, p) # D*1
    # pred = tf.matmul(tf.transpose(W), (o + unet)) # V*1

    with tf.Session() as sess:
        zeton = sess.run(m)
        import pdb; pdb.set_trace()
        print "y"

if __name__ == "__main__":
    network()
