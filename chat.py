import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('assurance_cleaned.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
import random

messages = [
    "Je n'ai pas bien compris. Pourriez-vous reformuler votre question, s'il vous plaît?",
    "Pour plus d'informations, veuillez contacter notre service.",
    "Nous vous remercions de contacter notre service et nous vous remercions pour votre compréhension."
]
def get_response(msg):
    global messages  # Pour accéder à la variable messages globale

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    # Si la probabilité prédite est inférieure à 0.75, retournez un message différent à chaque appel
    if messages:
        return messages.pop(0)
    else:
        # Réinitialisez la liste messages si tous les messages ont été renvoyés
       messages = [
          "Je suis désolé, je n'ai pas compris votre question. Pourriez-vous la reformuler, s'il vous plaît?",
          "Pouvez-vous reformuler votre question? Je ne suis pas sûr de comprendre.",
          "Je m'excuse, mais je n'ai pas saisi votre demande. Pourriez-vous la reformuler de manière différente?",
          "Votre question est un peu floue. Pourriez-vous fournir plus de détails, s'il vous plaît?",
          "Je suis ici pour vous aider, mais j'ai besoin d'une clarification. Pourriez-vous préciser votre question?",
          "Il semble y avoir un problème de compréhension. Pourriez-vous reformuler votre question de manière plus claire?",
          "Je ne suis pas certain de comprendre votre question. Pourriez-vous l'exprimer différemment?",
          "Excusez-moi, je ne suis pas sûr de ce que vous voulez dire. Pourriez-vous reformuler votre question?",
          "Je suis désolé, je ne parviens pas à comprendre votre question. Pourriez-vous la reformuler?",
          "Merci pour votre compréhension."
]
    


       return messages.pop(0)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)