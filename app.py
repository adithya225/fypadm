from flask import Flask, request, render_template
import joblib


app = Flask(__name__)
reg = joblib.load('admission_linearmodel.pkl')
sc1 = joblib.load('scaler1.pkl')
sc2 = joblib.load('scaler2.pkl')
sc3 = joblib.load('scaler3.pkl')
sc4 = joblib.load('scaler4.pkl')
sc5 = joblib.load('scaler5.pkl')

@app.route('/', methods=["GET","POST"])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = [str(x) for x in request.form.values()]
    gre = sc2.transform([[float(features[0])]])[0][0]/340
    print(gre)
    toefl = sc3.transform([[float(features[1])]])[0][0]/120
    ur = float(features[2])
    sop = float(features[3])
    lor = float(features[4])
    cgpa = (sc1.transform([[float(features[5])]])[0][0]*9.5)/100
    research = float(features[6])
    ssc = sc4.transform([[float(features[7])]])[0][0]/100
    hsc = sc5.transform([[float(features[8])]])[0][0]/100
    college_Location = features[9]

    final_features = []
    final_features.append(gre)
    final_features.append(toefl)
    final_features.append(ur)
    final_features.append(sop)
    final_features.append(lor)
    final_features.append(cgpa)
    final_features.append(research)
    final_features.append(ssc)
    final_features.append(hsc)
    print(final_features)
    prob = reg.predict([final_features])
    prob*=100
    print(prob)
    if college_Location=="USA":
        out = predictunivusa(prob)
        if out == "No admission For you":
            # out = reg.predict([[0.54039, 0.757186,5,4.0,4.5,0.304036,0,	0.303167,0.304036]])
            return render_template('predict2.html', prediction_text = out)
        else:
            return render_template('predict1.html', prediction_text = out)
    elif college_Location=="UK":
        out = predictunivuk(prob)
        if out == "No admission For you":
            return render_template('predict2.html', prediction_text = out)
        else:
            return render_template('predict1.html', prediction_text = out)
    elif college_Location=="CANADA":
        out = predictunivcanada(prob)
        if out == "No admission For you":
            return render_template('predict2.html', prediction_text = out)
        else:
            return render_template('predict1.html', prediction_text = out)
def predictunivusa(n):
    lsusa = ["American University, Washington, D.C.",
             "Georgetown University, Washington, D.C.",
             "George Washington University, Washington, D.C.",
             "Strayer University, Washington, D.C.",
             "University of Maryland, Washington, D.C.",
             "University of the Potomac, Washington, D.C.",
             "Delaware State University, Dover",
             "Central Connecticut State University, New Britain",
             "Fairfield University, Fairfield",
             "Quinnipiac University, New Haven",
             "Rensselaer at Hartford, Hartford",
             "Sacred Heart University, Fairfield",
             "Southern Connecticut State University, New Haven",
             "University of Connecticut, Mansfield",
             "University of Hartford, West Hartford",
             "University of New Haven, West Haven",
             "Western Connecticut State University, Danbury",
             "Yale University, New Haven"]
    k =""
    if n>=75 and n<=100:
        for i in range(75,98):
            if n>i and n<i+2:
                k+=lsusa[i-75]
                break
    else:
        k+="University of New Haven, West Haven"
    return k


def predictunivuk(n):
    lsuk = ["City University London, London",
            "Cranfield University, Cranfield",
            "Hult International Business School, London",
            "University of Leeds, Leeds",
            "London Business School, London",
            "University of Oxford, Oxford",
            "University of Cambridge, Cambridge",
            "University of Kent, Canterbury",
            "University of Manchester, Manchester",
            "University of Reading, Reading",
            "University of Warwick, Coventry"]
    k = ""

    if n >= 70 and n <= 100:
        for i in range(70, 98):
            if n > i and n < i + 3:
                k += lsuk[i - 50]
                break
    else:
        k += "No admission For you"
    return k


def predictunivcanada(n):
    lsca = ["Brock University, St. Catharines",
            "Cape Breton University, Sydney",
            "Carleton University, Ottawa",
            "Concordia University, Montreal",
            "Dalhousie University, Halifax",
            "HEC Montreal, Montreal",
            "Ivey Business School, Toronto",
            "Lakehead University, Thunder Bay",
            "McGill University, Montreal",
            "McMaster University, Hamilton",
            "Queens University School of Business, Kingston",
            "University of Ottawa, Ottawa",
            "University of Toronto, Toronto",
            "York University, Toronto"]

    k = ""

    if n >= 70 and n <= 100:
        for i in range(70, 98):
            if n > i and n < i + 2:
                k += lsca[i - 70]
                break
    else:
        k += "No admission For you"
    return k

if __name__=='__main__':
	app.run(debug=True)
