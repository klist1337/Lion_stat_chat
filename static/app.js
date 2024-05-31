class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }
        
        this.state = false;
        this.messages = [];
        
        // Appeler la méthode addInitialMessages dans le constructeur
        this.addInitialMessages();
        
        // Appeler la méthode display pour gérer les événements
        this.display();
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    addInitialMessages() {
        const initialMessages = [
            { name: "Sam", message: "Bonjour!" },
            { name: "Sam", message: "Bienvenue à Lion's Chat." },
            { name: "Sam", message: "Comment puis-je vous assister aujourd'hui ?" },
            { name: "System", message: "Voici quelques questions pour commencer:" }
        ];  

        // Ajouter les messages initiaux à la liste des messages
        this.messages = initialMessages;
        
        // Mettre à jour l'affichage du chat avec les messages initiaux
        this.updateChatText(this.args.chatBox);
        
        // Ajouter des boutons de questions initiales
        const initialQuestions = [
            "Renouvellement Demande",
            "Lyon stat by footbar",
            "Livraison Internationale",
            "integration appareils",
            "support client",
            "Politique Retour Echange",
            "Promotions Offres Speciales",
            "Avis Clients Produits",
            "La différence entre RC SPORT et AMO FRMF ?",
            "Où puis-je récupérer les feuilles de maladie ?",
            "Limitations Traitement",
            "Y a-t-il des limitations sur les types de traitement médical couverts par l'AMO FRMF ?",
            "Délais Remboursement",
            "Quels sont les taux de remboursement ?",
            "Quel est le quota pour le club ?",
            "Où puis-je envoyer mes demandes de remboursement ?",
            "Fondateurs Footbar",
            "Modes Paiement"
            // Ajoutez d'autres questions initiales ici
        ];
        
        // Choisir aléatoirement deux des messages initiaux
        const randomMessages = this.getRandomElements(initialQuestions, 2);
        
        // Créer et ajouter des boutons de questions initiales
        randomMessages.forEach(message => {
            const buttonHTML = `<button class='initial-question' data-message='${message}'>${message}</button>`;
            this.messages.splice(4, 0, { name: "System", message: buttonHTML });
        });
        
        // Mettre à jour l'affichage du chat avec les boutons de questions initiales
        this.updateChatText(this.args.chatBox);
        
        // Ajouter des gestionnaires d'événements pour les boutons de questions initiales
        const initialQuestionButtons = document.querySelectorAll('.initial-question');
        initialQuestionButtons.forEach(button => {
            button.addEventListener('click', () => {
                const message = button.getAttribute('data-message');
                this.sendMessage(message);
                
                // Supprimer le message "Voici quelques questions pour commencer :"
                this.messages = this.messages.filter(msg => msg.message !== "Voici quelques questions pour commencer:");
                this.updateChatText(this.args.chatBox);
            });
        });
        
    }

    sendMessage(message) {
        let msg = { name: "User", message: message };
        this.messages.push(msg);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: message }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            let msg = { name: "Sam", message: r.answer };
            this.messages.push(msg);
            this.updateChatText(this.args.chatBox);
        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(this.args.chatBox);
        });
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }

    // Fonction utilitaire pour obtenir des éléments aléatoires dans un tableau
    getRandomElements(arr, n) {
        var result = new Array(n),
            len = arr.length,
            taken = new Array(len);
        if (n > len)
            throw new RangeError("getRandom: more elements taken than available");
        while (n--) {
            var x = Math.floor(Math.random() * len);
            result[n] = arr[x in taken ? taken[x] : x];
            taken[x] = --len in taken ? taken[len] : len;
        }
        return result;
    }
}

const chatbox = new Chatbox();
