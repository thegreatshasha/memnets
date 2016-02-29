import tensorflow as tf

# porting the dataflow right now. we are using just a single hop and evaluating it on babl toy tasks

# dimensions
D = 3 # embedding space dimension
N = 2 # no of inputs/memories
V = 4 # vocabulary size

# data placeholder
x = 

# Define the embedding matrices as variable
A = tf.Variable() # input embedding matrix, V*D
B = tf.Variable() # questions embedding matrix,
C = tf.Variable() # input embedding matrix
W = tf.Variable() # prediction embedding matrix

# question embedding
u = tf.matmul(w,q)

# input embeddings
m = tf.matmul(A, x).swapaxes(0, 1)
c = tf.matmul(C, x).swapaxes(0, 1)

# positional and temporal encoding
# skipping this for now

# sum the elmements along horizontal axis for getting bag of words representation
cnet = c.sum(axis=2).transpose()
mnet = m.sum(axis=2).transpose()
unet = u.sum()

# get softmax probabilities for each memory with input
scores = np.dot(tf.transpose(mnet), unet)# get match score of u with each of the memorie
p = tf.nn.softmax(scores)

# get prediction
o = tf.matmul(cnet, p)
pred =
