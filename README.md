# Language-Detection-Model
Building a Language Detection model based on a data which has text and the language name using NLP and deeplearning.

Using vectorization and sklearn library I have build a basic model with logistic reggression which will detect which language is given as input.

Sklearn is very powerfull and very easy to build a model.

7 Languages used : ENGLISH, FRENCH, DUTCH, DANISH, SPANISH, ITALIAN, GERMAN

Language Detection.csv is the data,from that I extracted 7 languages data with labels and cleaned it using regex(Regular expression), preprocessing of data is done with the labels and i have given that data into a pipeline, pipeline is used for executing operations one after the other so i have given first operation as vectorization and second as the model training initialization.

Accuracy is 98%.
Thats it very easy and simple code.
Any doubts you can ask me on my git or email!.
Feel free.

Note : Running the notebook on colab will generate a model which is not upto date version of sklearn so you can run modelbuilding.py file to build your model for local model building.
