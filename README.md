# KaggleSpeechChallenge
TensorFlow Speech Recognition Challenge
Classify audio in .wav format. 


## Location
Orignally found at :
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

Classificatoin of Audio data.

## Results of this approach
My final score 0.899 with the Kernel uploaded here. Final rank was in the top 4%

The final approach is an ensemble model with an RNN at its core. Supplementary Conv2D net feeds more data after being trained seperately.

![alt tag](http://ernstthuer.eu/tensorboard_RNN.png "Graph")

## Caveat
The main limiting factor was that the Challenge offered Cloud computing Credits to competitors,Not available to me, since I signed up late. Created a model to fit into my local gpu ( < 3gb). Distance to the competition winner is ~1 %, 


## Additional information
In the spirit of learning, I designed the whole pipeline in Tensorflow native via the python API.

The challenge was posed by GoogleBrain.

preprocessing ipython notebooks and the classes used for estimating the hyperparameters are in the corresponding folders
