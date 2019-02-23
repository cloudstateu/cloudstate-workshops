
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

import azureml.core
from azureml.core import Workspace, Experiment, Run

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)


# In[2]:


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
image_index = 0 # take first test image
img = mnist.test.images[image_index]

# draw image
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()


# In[3]:


# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')


# In[4]:


experiment_name = 'tensorflow-mnist'

experiment = Experiment(workspace=ws, name=experiment_name)


# In[5]:


run = experiment.start_logging()

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# prepare dir for output model
os.makedirs('./outputs/model', exist_ok=True)
saver = tf.train.Saver()

run.log("Total training epochs: ", training_epochs)
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            run.log("Cost", avg_cost)

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    classification = sess.run(tf.argmax(pred, 1), feed_dict={x: [img]})
    print(classification[0])
    
    saver.save(sess, './outputs/model/mnist-tf.model')
    for f in os.listdir('./outputs/model'):
        if f.startswith("mnist") or f.startswith("checkpoint"):
            run.upload_file(name = 'outputs/model/' + f, path_or_stream = 'outputs/model/' + f)
run.complete()


# In[6]:


run


# In[ ]:


get_ipython().system(u'az login')
get_ipython().system(u'az provider show -n Microsoft.ContainerInstance -o table')


# In[ ]:


get_ipython().system(u'az provider register -n Microsoft.ContainerInstance')


# In[7]:


model = run.register_model(model_name='tf-dnn-mnist', model_path='outputs/model')


# In[8]:


get_ipython().run_cell_magic(u'writefile', u'score.py', u'import tensorflow as tf\nimport numpy as np\nimport os\nimport json\nfrom azureml.core.model import Model\n\ndef init():\n    global w_out, b_out, sess\n    tf.reset_default_graph()\n    model_root = Model.get_model_path(\'tf-dnn-mnist\')\n    graph = tf.train.import_meta_graph(os.path.join(model_root,\'mnist-tf.model.meta\'))\n    \n    sess = tf.Session()\n    graph.restore(sess, os.path.join(model_root, \'mnist-tf.model\'))\n    w_out, b_out = sess.run(["W:0", "b:0"])\n\ndef run(raw_data):\n    img = np.array(json.loads(raw_data)[\'data\'])\n    x_out = tf.placeholder(tf.float32, [None, 784]) \n    predictor = tf.nn.softmax(tf.matmul(x_out, w_out) + b_out)   \n\n    classification = sess.run(tf.argmax(predictor, 1), feed_dict={x_out: [img]})\n    return str(classification[0])')


# In[9]:


from azureml.core.runconfig import CondaDependencies

cd = CondaDependencies.create()
cd.add_conda_package('numpy')
cd.add_tensorflow_conda_package()
cd.save_to_file(base_directory='./', conda_file_path='myenv.yml')

print(cd.serialize_to_string())


# In[10]:


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={'name':'mnist', 'framework': 'TensorFlow'},
                                               description='Tensorflow on MNIST')


# In[11]:


from azureml.core.image import ContainerImage

imgconfig = ContainerImage.image_configuration(execution_script="score.py", 
                                               runtime="python", 
                                               conda_file="myenv.yml")


# In[12]:


get_ipython().run_cell_magic(u'time', u'', u"from azureml.core.webservice import Webservice\n\nservice = Webservice.deploy_from_model(workspace=ws,\n                                       name='tf-mnist-svc10',\n                                       deployment_config=aciconfig,\n                                       models=[model],\n                                       image_config=imgconfig)\n\nservice.wait_for_deployment(show_output=True)")


# In[13]:


import json

test_samples = json.dumps({"data": img.tolist()})
test_samples = bytes(test_samples, encoding='utf8')

result = service.run(input_data=test_samples)
print(result)


# In[ ]:


from azureml.core.model import Model
# image_index = 1 # take first test image
# img = mnist.test.images[image_index]

# draw image
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.show()

tf.reset_default_graph()

# load model
x_out = tf.placeholder(tf.float32, [None, 784]) 

#model_root = Model.get_model_path('tf-dnn-mnist', _workspace=ws)
#print(model_root)
#graph = tf.train.import_meta_graph(os.path.join(model_root, 'mnist-tf.model.meta'))

graph = tf.train.import_meta_graph('./outputs/model2/mnist-tf.model.meta')

# for tensor in tf.get_default_graph().get_operations():
#     print (tensor.name)

with tf.Session() as sess:
    # restore the saved vairable
    graph.restore(sess, './outputs/model2/mnist-tf.model')
    #graph.restore(sess, os.path.join(model_root, 'mnist-tf.model'))
    w_out, b_out = sess.run(["W:0", "b:0"])
    print(b_out)
    predictor = tf.nn.softmax(tf.matmul(x_out, w_out) + b_out)     
    classification = sess.run(tf.argmax(predictor, 1), feed_dict={x_out: [img]})
    
    print("Predicted value is: " + str(classification[0]))
    


# In[ ]:


get_ipython().system(u'ls -al ./outputs/')


# In[ ]:


get_ipython().system(u'ls -al ./outputs/model')


# In[ ]:


get_ipython().system(u'rm ./outputs/checkpoint')

