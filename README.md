# Speech Emotion Recognition
![image](https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/9afa4ff1-d2f3-4702-a7c3-bf1e2337ffb1)

Emotion is an quality most people associate with human beings. Paired with speech, emotions allow people to communicate and articulate their feelings. 

Speech Emotion Recognition (SER) is the task of speech processing that aims to recognize and categorize the emotions expressed in spoken language. The goal is to determine the emotional state of a speaker, such as happiness, anger, or sadness from their speech patterns such as loudness, pitch, rhythm, and tone. 

Understanding human emotions paves the way to understanding people's needs better and, ultimately, providing better service. 

## Business Problem
We've all personally experienced the frustration of calling into call centers. You're often stuck on an automated menu that can't seem to direct you to an agent in the proper division because computers are far from good enough with language to replace a customer service agent. Other times you manage to reach an actual agent who at times is less than helpful. This is where utilizing SER technology in Verizon call centers can greatly improve customer satisfaction and reduce churn rate.

Verizon call center employees are conducting over 60 calls a day. While humans can notice things like if the customer is speaking more quickly, if the caller is silent for a long time or if the caller and agent are talking over one another, its a struggle to do so consistently when they may be tired from the call volumes they experience. Other times they may feel they understand the customers feelings only to realize they were completely off the mark. SER can take the guess work out of customer calls and allow employees to be as responsive as possible to customers' needs.

The technology can detect a customers emotional state from the automated menu and automatically transfer customers with negative emotional state to a live agent or prioritize those with such emotions in the live queue.

Also, SER can be used to detect acoustic cues to recognize hate speech and discrimination and automatically end the call to protect employees from verbal abuse.

Finally, SER can help expand on existing databases collected from recording customer calls for employee training to better train employees on how to respond to customers experiencing dissatisfaction.

## Data Understanding
CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) is a data set consisting of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities. Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

https://github.com/CheyneyComputerScience/CREMA-D


Class Distribution

<img width="454" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/583d759f-a352-40b4-b5cd-9361190cff97">


Waveplot and Spectrogram of Angry Emotion   

<img width="504" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/fa7ed958-57d1-4352-9ae0-c23544d49ea7">


Waveplot and Spectrogram of Sad Emotion   

<img width="507" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/bf7382f2-adbd-4b38-9a05-b4627c70e5f3">

## Identifying and Extracting Features

Three approaches:

After an initial load of an audio file: x , sr = librosa.load(audio_file) :

* Numeric Features extraction: From the librosa sound arrays, extracted features including Zero Crossing Rate, Mel-Frequency Cepstral Coefficients, Root Mean Square, Mel-Spectrogram, Chromogram, Spectral Centroid, Spectral Bandwidth, Spectral Contrast, Spectral Rolloff, and Tonnetz. Only the mean, standard deviation, kurtosis and skew of each feature were kept for model training of Linear Regression, Random Forest Classifier, Gradient Boosting Classifier, KNeighbors Classifier, and SVC models.

* Spectrograms generation: Instead of extracting numeric features, generated a spectrogram image of a sound. Such spectrogram visualizes the signal strength over time at various frequencies. A CNN model can be trained directly using images/spectrograms.

* Hugging Face Wav2Vec Feature Extractor: Wav2Vec is an approach for speech recognition by learning representations of raw audio. In general, we use mfcc features to train speech recognition systems. Instead, if we use the encoding or representation of Wav2Vec we can achieve almost similar results while using less labeled training data.

## Modeling

The best model using the numeric feature method was the XGBClassifier which returned a 48% accuracy overall on the test set. It seemed to have the most trouble confusing happy and angry emotions, disgust with neutral emotions and sadness with fear.

XGBClassifier Classification Report


<img width="281" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/c6fd1f14-f8c5-4739-8999-2566ed6e233b">


XGBClassifier Confusion Matrix


<img width="527" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/77b6e661-050f-4b2f-bdb4-b7b7ed74d2c7">

The best model using the spectrogram feature method was the MobileNet Convolution Neural Network model with two added dense layers and unfreezing of the last convolutional layer of the pretrained model. The model performed similarly at 50% accuracy on the test set. Similarly, it confused anger with happiness and fear with sadness.


MobileNet Classification Report


<img width="311" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/a2e1adee-0300-463e-a221-7faa82738303">

MobileNet Confusion Matrix


<img width="521" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/1e47fb75-1a5c-4e92-8030-cc8b8a485888">

The TFWav2Vec model increased the accuracy substantially to 72%. The final model  had the hardest time confusing fear and sadness, which was seen in the other model iterations.

TFWav2Vec Classification Report


<img width="302" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/0e6ba140-4d3d-47d5-a72e-eb38d7599b5d">

TFWav2Vec Confusion Matrix


<img width="513" alt="image" src="https://github.com/biannagas/Speech-Emotion-Recognition/assets/131709766/bad3e5b1-c34c-4c60-9e0b-cc9183d113fa">

# Next Steps
* Instead of taking the mean, std, kurtosis and skew of the extracted numeric features, try keeping all of the feature information from each window to feed into the classification model.

* Use other Hugging Face pretrained models like the Audio Spectrogram Transformer model in image classification.

* Use video clips to detect emotion.

* Use a different datasets or combination of emotion recognition datasets like the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), Toronto emotional speech set (Tess), or Surrey Audio-Visual Expressed Emotion (Savee).


















