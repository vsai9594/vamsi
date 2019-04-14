from flask import Flask, request, render_template, url_for, session, redirect, g
from hi import hi
import os
app = Flask(__name__)
@app.route('/getPredection',methods=['GET' , 'POST'])
def getPredection():
    inp = request.form
    predict1=inp['quality']
    predict2 = inp['living_area']
    predict3 = inp['car_size']
    predict4 = inp['garage_area']
    predict5 = inp['first_floor_area']
    predict6 = inp['bathroom']
    predict7 = inp['rooms']
    predict8= inp['year']
    price=""
    try:
        price=10
    except:
        print("unable to predict price")
    if price:
        return render_template("base.html",results=price)
    else:
        return render_template("base.html", results=False)




