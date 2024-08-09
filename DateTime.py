# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:13:02 2023




@author: USER
"""

from datetime import datetime
#datetime object containing current ate and time
Now = datetime.now()
print(Now)

discount_rate = 0.10
tax_rate = 0.06

subtotal = float(input("Enter the subtotal: "))
Week = Now.weekday()
print(Week)
