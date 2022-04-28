# RZ-V2L-FaceRecognition-demo

#### BSP-version

BSP: RZV2L/RZV2L-SMARC-EVK/Version: 1.0

#### SDK

SDK compilation requires extra library：opencv dnn

#### Save face feature vector and program exit
press 'S' or 's' to save face feature vector. Press the button once to save the face feature vector in the current frame. When there are multiple faces, only the last face feature vector is saved.

Face feature vector save path: exe/face_vectors/person***.bin.

Save file naming method: person+*(*represents the number of face feature vectors),If you want to display the real name, you can change the corresponding face feature vector file to the real name, such as eric.bin.

press 'Q' or 'q' to exit program

#### Demo effect

<video src="https://github.com/Eric036/RZ-V2L-FaceRecognition-demo/demo.mp4"></video>
