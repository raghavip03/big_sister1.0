Project Name:
big_sister1.0

Project Description:
big_sister1.0 is a chatbot designed to assist users with college planning. It provides guidance on various aspects of the college application process, including choosing the right college, creating a strong application, and general tips for planning a successful college journey. This chatbot is specifically tailored to support students on visas, such as H4 and F1, who often have limited resources when it comes to selecting colleges that align with their unique circumstances.

Technical Components:
  0. Designing Intents:
    - Designed a JSON representation of 'intents' the user has when asking questions
    - 'intents' are categorized into tags, each tag having 'patterns' of similar questions
    and pre-determined 'responses'.
  1. Data Processing
    - Utilizes the 'nltk' library to tokenize/break-up the input sentences and
    stem words to their root form
    - Implemented a bag_of_words model that converts tokenized sentences into
    fixed numerical representation of the words (1 or 0) which shows either the
    presence or absence of the tokenized words in each pattern.
  2. Neural Network:
  - The core of the chatbot is a feedforward neural network with two hidden layers, built using PyTorch.
  - The network is trained to classify input sentences into predefined intent categories using the       CrossEntropyLoss function and the Adam optimizer.
  - Training data, including tokenized words and corresponding tags, are loaded from a JSON file (intents.json).

  3. Model Training and Saving:
  - The model is trained over several epochs to minimize the loss between predicted and actual tags.
  - After training, the model's state, along with relevant metadata (input size, output size, etc.), is saved to a file (model_data.pth) for later use.

  4. Inference:
  - During inference, the chatbot tokenizes user input, converts it into a bag-of-words vector, and feeds it into the trained model to predict the user's intent.
  - Based on the predicted intent, the chatbot selects and returns an appropriate response from the predefined responses in the intents.json file.

  5. Response Generation:
  - The chatbot's responses are generated based on the intent with the highest probability. If the model is confident (probability > 0.75), it selects a response; otherwise, it indicates that it doesn't understand the query.

Installation and Dependencies:
  1. Clone the Repository:
    git clone https://github.com/raghavip03/big_sister1.0.git
  2. Install Dependencies:
    pip install -r requirements.txt
  3. Train the model:
    python train.py
  4. Run the chatbot:
    python big_sister.py

7. Future Enhancements
  - Improving Accuracy: Experiment with different neural network architectures and hyperparameters to enhance the model's accuracy.
  - Expanding Intents: Add more intents and patterns to cover a broader range of user queries.
  - User Interface: Develop a web or mobile interface for easier interaction with the chatbot.

snkfjsjdlsd