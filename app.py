from flask import Flask,render_template,request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string

app=Flask(__name__)
nltk.data.path.append('./nltk_data')
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
ps=PorterStemmer()


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()  
    for i in text:
        y.append(ps.stem(i))  
    return " ".join(y)

def predict_spam(message):
    transformed_sms=transform_text(message)
    # tf=pickle.load(open("vectorizer.pkl","rb"))
    # model=pickle.load(open("model.pkl","rb"))
    vector_input=tf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    return result

# print(predict_spam("CONGRATULATIONS! You have been randomly selected to receive a FREE REWARD. Click here to claim your prize: [FAKE-LINK-DO-NOT-CLICK]Offer expires in 24 hours!"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html', result=result)  # Pass 'result' to the template

if __name__=="__main__":
    tf=pickle.load(open("vectorizer.pkl","rb"))
    model=pickle.load(open("model.pkl","rb"))
    app.run(host="0.0.0.0")


    
    