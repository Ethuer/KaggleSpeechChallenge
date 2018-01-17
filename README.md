# KaggleSpeechChallenge
TensorFlow Speech Recognition Challenge

## Location
Orignally found at :
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

Classificatoin of Audio data.

## Results of this approach
My final score 0.899 with the Kernel uploaded here.

The final approach is an ensemble model with an RNN at its core. Supplementary Conv2D net feeds more data after being trained seperately.

## Caveat
The main limiting factor was that the Challenge offered Cloud computing Credits to competitors, which I missed since I signed up late. So my model fit into my local gpu ( < 3gb),  considering that, distance to the competition winner is < 1.2 %, So I am uploading my notebooks anyways.


## Additional information
In the spirit of learning, I designed the whole pipeline in Tensorflow native via the python API.

It was a challenge posed by GoogleBrain.


