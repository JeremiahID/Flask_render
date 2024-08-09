from flask import Flask, redirect, url_for, render_template, flash, session, logging, request
import sqlite3
import mysql.connector
from wtforms import Form,StringField,TextAreaField, PasswordField, validators, RadioField, SubmitField
from wtforms.validators import DataRequired
from passlib.hash import sha256_crypt

app = Flask(__name__)

  #connect to database
con = sqlite3.connect("web.db")
cursor= con.cursor()

#class for home
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

#class for check up
@app.route("/Test" , methods = ["POST" ,"GET"])
def Test():
    return render_template("Test.html")

#class for sign-up
@app.route("/Sign-in" , methods = ["POST", "GET"])
def sign_in():
     if request.method == "POST":
            Email = request.form["email"]
            password = request.form["password"]
            con=sqlite3.connect("web.db")
#           cursor = con.cursor()
#           statement = f"SELECT * from web  WHERE email ='{Email}', Password = '{password}';"
#           con.execute(statement)
            if not con.fetchone():
                return render_template("Sign-in.html")
            else:
                return render_template ("Test.html")
     else:
            request.method == "GET"
            return render_template("Sign-in.html")
            


#class for registration
class RegisterForm(Form):
    Email = StringField( "email", validators =[DataRequired()])
    FirstName = StringField("fname", validators =[DataRequired()])
    LastName = StringField("lname", validators =[DataRequired()])
    Password =PasswordField (  'password',validators =[DataRequired()])
    R_assword = PasswordField( "cpassword")
    gender = RadioField( "Gender" , choices= ["Male", "Female"])
    submit = SubmitField( "Send")

#app route for resgisration
@app.route("/Register", methods = ["POST" , "GET"])
def Reg():
    form = RegisterForm(request.form)
    con=sqlite3.connect("web.db")
    cursor= con.cursor()
    if request.method == "POST":
        if(request.form["email"]!="" and request.form["Password"]!="" and request.form["Re_password"]!="" and request.form["First_Name"]!="" and request.form["Last_Name"]!="" and request.form["Gender"]!=""):
            Email = request.form.get["email"]
            password = request.form["password"]
            c_password = request.form["cpassword"]
            FirstName = request.form["fname"]
            LastName  = request.form[" lname "]
            Gender = request.form["Gender"]
            statement = f"SELECT * from web  WHERE email ='{Email}', Password = '{password}', Re_password ='{c_password}', First_Name = '{FirstName }', Last_Name  ='{LastName }', Gender = '{Gender}';"
            con.execute(statement)
            data = con.fetchone()
            if data:
                return render_template("error.html")
            else:
                if not data:
                    con.execute("INSERT INTO web (email, Password, Re_password, First_Name, Last_Name, Gender) VALUES (?,?,?,?,?,?)", (Email, password, c_password, FirstName, LastName, Gender))
                    con.commit
                    con.close
                return render_template("Sign-in.html")

    elif request.method == "GET":
        return render_template("Register.html", form=form)



if __name__ == "__main__" :
    app.run(debug=True) 