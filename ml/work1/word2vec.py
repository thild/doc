
# coding: utf-8

# In[25]:


import gzip
import gensim 
import logging
import numpy
logging.basicConfig(level=logging.DEBUG)


# In[3]:


def read_input(input_file):
	"""This method reads the input file which is in gzip format"""
	
	logging.info("reading file {0}...this may take a while".format(input_file))
	
	with gzip.open (input_file, 'rb') as f:
		for i, line in enumerate (f): 

			if (i%10000==0):
				logging.info ("read {0} reviews".format (i))
			# do some pre-processing and return a list of words for each review text
			yield gensim.utils.simple_preprocess (line)


# In[4]:


a=10
print (a)


# In[5]:


documents = list (read_input ("reviews_data.txt.gz"))
print ("Done reading data file")


# In[7]:


model = gensim.models.Word2Vec (documents, size=100, window=10, min_count=2, workers=10)
print ("Training model")
model.train(documents,total_examples=len(documents),epochs=10)
print ("Model trained")


# In[9]:


w1 = "dirty"
print (model.wv.most_similar (positive=w1))


# In[10]:


print (model['dirty'])


# In[11]:


w1 = "france"
print (model.wv.most_similar (positive=w1,topn=6))


# In[12]:


w1 = "dirty"
print (model.wv.most_similar (positive=w1))


# In[13]:


w1 = "dirty"
w2  ="clean"
print (model.wv.similarity (w1,w2))


# In[14]:


w1 = "red"
print (model.wv.most_similar (positive=w1,topn=6))


# In[15]:


import pickle


# In[16]:


fileObject = open("modelogensim",'wb') 


# In[17]:


pickle.dump(model,fileObject) 


# In[18]:


fileObject.close()


# In[20]:


fileObject = open("modelogensim",'rb')  
# load the object from the file into var b
b = pickle.load(fileObject)


# In[21]:


print (b.wv.most_similar (positive=w1,topn=6))


# In[22]:


a = b.wv.most_similar(positive=w1,topn=6)


# In[24]:


a[1][1]

