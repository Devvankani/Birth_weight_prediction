from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def clean(data):
    gest = float(data["gestation"])
    p = int(data["parity"])
    ag = float(data["age"])
    hi = float(data["height"])
    wi = float(data["weight"])
    si = float(data["smoke"])
    
    clean_data = {
        "gestation": [gest],
        "parity": [p],
        "age": [ag],
        "height": [hi],
        "weight": [wi],
        "smoke": [si]      
    }
    
    return clean_data

@app.route("/pred",methods=["POST"])
def pred_model():
    data = request.get_json()
    clean_data = clean(data)
    test_data = pd.DataFrame(clean_data)
    
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
        
    pred = model.predict(test_data)
    res =round(float(pred[0]),2)
    
    
    return jsonify({"response": res})
    


if(__name__ == "__main__"):
    app.run(debug=True)

    
    