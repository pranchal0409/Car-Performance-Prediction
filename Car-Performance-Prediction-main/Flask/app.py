from flask import Flask , render_template , request
import pickle
app = Flask(__name__)
model = pickle.load(open('regression.pkl','rb'))

@app.route('/')
def intro():
    return render_template('index.html')
@app.route('/login', methods = ["POST"])
def login():
    
    cyl = request.form["cyl"]
    dis = request.form["dis"]
    hp = request.form["hp"]
    w = request.form["w"]
    a = request.form["a"]
    my = request.form["my"]
    ori = request.form["ori"]
    total = [[int(cyl),int(dis),int(hp),int(w),int(a),int(my),int(ori)]]
    
    p = model.predict(total)
    p =p[0]
    return render_template('index.html',label = "The mileage(kmhr) of the car is  "+str(p))

if __name__=='__main__':
    app.run(debug = True,port = 9000)
