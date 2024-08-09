# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:09:03 2022

@author: USER
"""
import sqlite3

  #connect to database
con = sqlite3.connect("web.db")
cursor = con.cursor()


#con.execute( ''' INSERT INTO Sign_Up ('email', 'Password','Re_password', 'First_Name', 'Last_Name', 'Gender') 
 #           values ("jeremiahefe.j@gmail.com", "Jerry1234","Jerry1234", "Jeremia Efe", "Idahosa", "male") ''')
print("succesfully inserted")
result = con.execute("SELECT * FROM Sign_Up ")          
for data in result:
    print('email: ', data[0])
    print('Password: ', data[1])
con.commit
      
con.commit
