### This is the main thing
######
from __future__ import division
import numpy
import gurobipy as GRB
import matplotlib as mat
import math as math
import pylab as pl
import csv
import sys
import operator
import time
from scipy.stats import truncnorm
import collections
from bisect import bisect_left



start_time = time.time()




DataReader = False
DataReaderFile = "Final-4-3-2-1-NoFlatSalv-Income189to191.csv"
TmaxTemp = 189


# Parameters

# Battery Cap is reported in kWh
Battery_Cap = 13.5
New_Battery_Cost= 620.0
Demand = 15

# For the steps, consider we have 10^2 already so take 2 zeroes off
# Decimals + 1 = the precision you want, 1 -> 1+1 = 1/10^2 precision
Decimals = 0
ellDecimals = 3
Battery_Cap_Fixer = 0
hDecimals = 1
Battery_Cap_Steps = 1


# Shows How many hours we need to generate full charge for battery (How big is the panel)
PV_Power_Output= Demand/4




A1s={}
A0s={}

ell=[]
h=[]

# Temperature
tau=25

# For half a day, %2.5 annual interest rate 
lamda= 0.99997

Peak_Sulight_Hours=[]
a1_Step_Size=2
a0_Step_Size=2

mu= 10.35#1.035
Tmax=20
phi = 1
Minimum_Decrease = 0.01
ell_Step_Size = 4
ell_Min_Cap = 75
# Variables
Income={}
SalvageIncome={}
phi = -1
BigM= 99999999999999999999999

print "Price",mu,"H Steps",Battery_Cap_Steps


# Income function

Months = [31,28,31,30,31,30,31,31,30,31,30,31]
mMin=[2.1,2.8,4,4.3,4.9,5.2,5,5,4.4,3.8,2.2,2.2]
mMax=[3.5,4.4,5.6,6.8,6.5,7.1,6.6,6.2,6,5.6,4,3.5]
mAvg=[2.9,3.6,4.6,5.4,5.7,6,5.8,5.6,5,4.5,3.2,2.7]
DaysOfAll= []

# for i in range(Months[0]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[0], mMax[0], size=10))
# for i in range(Months[1]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[1], mMax[1], size=10))
# for i in range(Months[2]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[2], mMax[2], size=10))
# for i in range(Months[3]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[3], mMax[3], size=10))
# for i in range(Months[4]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[4], mMax[4], size=10))
# for i in range(Months[5]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[5], mMax[5], size=10))
# for i in range(Months[6]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[6], mMax[6], size=10))
# for i in range(Months[7]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[7], mMax[7], size=10))
# for i in range(Months[8]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[8], mMax[8], size=10))
# for i in range(Months[9]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[9], mMax[9], size=10))
# for i in range(Months[10]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[10], mMax[10], size=10))
# for i in range(Months[11]):
# 	DaysOfAll.append(truncnorm.rvs(mMin[11], mMax[11], size=10))

# for j in range(len(Months)):
# 	for i in range(Months[j]):
# 		DaysOfAll.append(truncnorm.rvs(mMin[0], mMax[0], size=10))



# for i in range(Tmax):
# 	if i <= Tmax/4:
# 		DaysOfAll.append([4])
# 	elif i >= Tmax/4 and i <= Tmax/2:
# 		DaysOfAll.append([3])
# 	elif i >= Tmax/2 and i <= 3*Tmax/4:
# 		DaysOfAll.append([2])
# 	elif i >= 3*Tmax/4:
# 		DaysOfAll.append([1])


# DaysOfAll = [[2]]*361
# DaysOfAll.extend([[4]]*4)

# DaysOfAll = [[2]]*182
# DaysOfAll.extend([[2]]*183)
DaysOfAll = [[4]]*10
DaysOfAll.extend([[2]]*10)

if DataReader == True:
	# 0 74000   0   0   ::::::: 27880.21714 Night   100000  40  0   0
	# 0 75000   0   1   ::::::: 28313.37825 Replace 100000  40  0   -   -
	# 0 75001   0   1   ::::::: 27801.75806 Keep    74998   135 0   135 28313.37825
	#TmaxTemp = Tmax
	with open(DataReaderFile) as csvfile:
	        readCSV = csv.reader(csvfile, delimiter=',')
	        for row in readCSV:
	        	#if int(row[3]) == 0:
	        	Income[int(row[0]),int(row[1]),int(row[2]),int(row[3])] = [float(row[5]),row[6],float(row[7]),row[8],row[9],row[10],row[11]]
	        	#else:
	        	#	Income[int(row[0]),int(row[1]),int(row[2]),int(row[3])] = [float(row[5]),row[6],float(row[7]),row[8],row[9],row[10],row[11]]
	        # if int(row[0]) < TmaxTemp:
	        # 	TmaxTemp = int(row[0])
	Tmax = TmaxTemp
	print "Read from previous files and starting at day",Tmax 

    
# During day

# Replace cost
def R(ell,h):
    return New_Battery_Cost

# Keep cost
def K(ell,h):
    return 0
    

# Power output pl.frange in t    
def Available_Power_Output(t,ell,h,a1_Step_Size):
    Maximum_Y_Available=min(PV_Power_Output*Peak_Sulight_Hours[t]/ell,1)
    output=range(0,Maximum_Y_available,a1_Step_Size)
    return output


# Demand at night t
def Nightly_Demand(t):
	return [1.5,1,3,2,2.5,1.25]

def Demand_Probability(t):
    return [1/(len(Nightly_Demand(t)))]*(len(Nightly_Demand(t)))


def Salvage_Value(ell,h):
	# if ell == 1:
	# 	return New_Battery_Cost *.7
	# elif ell <= ell_Min_Cap:
	# 	return New_Battery_Cost*.3
	# else:
	# 	return New_Battery_Cost * .7* ell
	return 100
	

# Nightly income
def Night_Cost(t,a0,demand):
	return max(Demand - a0,0) * mu 
	#return GRB.quicksum(max(demand[i] - a0,0) * mu * Demand_Probability(t)[i] for i in range(len(demand))).getValue()

def Maximum_Sunlight(t):
	if t+1 > 365 :
		a = t % 365
	else:
		a = t

	return DaysOfAll[a]


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before



alpha_sei = 0.0575
beta_sei = 121
K_Delta_1 = 140000
K_Delta_2 = -0.501
K_Delta_3 = -123000
K_sigma = 1.04
Sigma_ref = 0.5
K_T = 0.0693
T_ref = 25
K_t = 0.000000000414 * 12 * 3600

tau = 25
S_t = K_t
S_T = math.exp(K_T * (tau - T_ref)*(T_ref/tau))

def Degradation_Fucntion(ell_Current,h,tau,a0,a1):
	S_sigma = math.exp(K_sigma*(a1 - Sigma_ref)) 
	if a0 !=0:
		S_delta = math.pow((K_Delta_1 * math.pow(a0,K_Delta_2) + K_Delta_3),-1)
	else:
		S_delta = 0
	F_T_D = (S_delta+S_t)*S_sigma*S_T
	#print F_T_D,"F_T_D","S_sigma",S_sigma,S_t,S_T
	# if ell_Current == 1:
	# 	#print '1',alpha_sei * math.exp(-beta_sei*F_T_D),'2', (1-alpha_sei)*math.exp(-F_T_D)
	# 	return (alpha_sei * math.exp(-beta_sei*F_T_D) + (1-alpha_sei)*math.exp(-F_T_D))
	# else:
	return (ell_Current)* math.exp(-(F_T_D))
    


# def Mrange(x, y, jump):
#   while x <= y:
#     yield x
#     x += jump


# def a1_Action_Set(a1_Chosen,t):
#     return list(Mrange(0,a1_Chosen,a1_Step_Size))


# def a1_Probability_set(A1):
# 	return[1/len(A1)]*len(A1)



# def a0_Action_Set(h,ell,tau,t):
# 	return pl.frange(0,h,a0_Step_Size)

ellrange= range(ell_Min_Cap* pow(10,ellDecimals),100*pow(10,ellDecimals)+1)

hrange=range(0,int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)+Battery_Cap_Steps),Battery_Cap_Steps)

sellrange=range((ell_Min_Cap - 1)* pow(10,ellDecimals),100*pow(10,ellDecimals)+1)

Demand = Demand*pow(10,Battery_Cap_Fixer+1)
Best_Income={}
tcurrent = Tmax
for t in range(Tmax,-1,-1):
	print 'Time Step ',t
	print("--- %s seconds ---" % (time.time() - start_time))
	if t == Tmax and DataReader == False:
		phi = 0

		for i in (sellrange):
			ell = i/pow(10,ellDecimals)/100
			hrange = range(0,int(round(Battery_Cap*ell,hDecimals)*pow(10,Battery_Cap_Fixer+1))+int(Battery_Cap_Steps/2),Battery_Cap_Steps)
			for j in (hrange):
				h = j/(Battery_Cap*pow(10,Battery_Cap_Fixer+1))
		 		Income[t,i,j,1]=[-Salvage_Value(ell,h),"Salvage","-","-","-","-",'-']
		continue

	else:
		if DataReader == True and t == Tmax:
			print "Starting from previous data at day",Tmax
			phi = 1
			TempIncome = {}
		if phi == 0:
			print "Phi 0"
			for i in sellrange:
				ell = i/pow(10,ellDecimals)/100
				hrange = range(0,int(round(Battery_Cap*ell,hDecimals)*pow(10,Battery_Cap_Fixer+1))+int(Battery_Cap_Steps/2),Battery_Cap_Steps)


				for j in hrange:
					TempIncome={}
					h = min(j/(Battery_Cap*ell*pow(10,Battery_Cap_Fixer+1)),1)
					a0 = h#round(min(Demand,j/pow(10,Battery_Cap_Fixer+1))/(ell*Battery_Cap),Decimals)
					a00 = j#h * Battery_Cap*ell

					#aa0 = int(math.floor((min(Demand,h*Battery_Cap*ell)/(ell*Battery_Cap))*100*pow(10,Decimals)))
					#NewEll = int(math.floor(Degradation_Fucntion(ell,h,tau,a0,0)*100*pow(10,ellDecimals)))
					NewEll = math.floor(round(Degradation_Fucntion(ell,h,tau,a0,0),ellDecimals+2)*pow(10,ellDecimals+2))/pow(10,ellDecimals+2)
					if NewEll <= ell_Min_Cap/100:
						Income[t,i,j,0] = [Night_Cost(t,a00,Demand)+ lamda * Income[t+1,int(ell_Min_Cap*pow(10,ellDecimals)),min(int((h - a0)*(Battery_Cap*NewEll*pow(10,Battery_Cap_Fixer+1))),int(round(Battery_Cap*NewEll,hDecimals)*pow(10,Battery_Cap_Fixer+1))),1][0],"Night",100*pow(10,ellDecimals),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),t,a0*ell*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),'-']

					else:
						#Income[t,i,j,0] = [Night_Cost(t,a00,Demand)+ lamda * Income[t+1,int(NewEll*pow(10,ellDecimals+2)),int((h - a0)*(Battery_Cap*pow(10,Battery_Cap_Fixer+1))),1][0],"Night",NewEll*pow(10,ellDecimals+2),(h-a0)*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),t,a0*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),'-']
						Income[t,i,j,0] = [Night_Cost(t,a00,Demand)+ lamda * Income[t+1,int(NewEll*pow(10,ellDecimals+2)),min(int((h - a0)*NewEll*(Battery_Cap*pow(10,Battery_Cap_Fixer+1))),int(Battery_Cap*NewEll*pow(10,Battery_Cap_Fixer+1))),1][0],"Night",NewEll*pow(10,ellDecimals+2),(h-a0)*(Battery_Cap*ell*pow(10,Battery_Cap_Fixer+1)),t,a0*(Battery_Cap*ell*pow(10,Battery_Cap_Fixer+1)),'-']

					#A0s[t,i,j,0]=a0
					#print Income[t,i,j,0]
					# print t,i,j,Income[t,i,j,0]
						# if t == 363:
						# 	print t,i,j,Income[t,i,j,0] 
						# 	print Night_Cost(t,a00,Demand),"new ell", int(NewEll*pow(10,ellDecimals+2)),"New h",min(int((h - a0)*(Battery_Cap*pow(10,Battery_Cap_Fixer+1))),int(round(Battery_Cap*NewEll,hDecimals)*pow(10,Battery_Cap_Fixer+1))) ,Income[t+1,int(NewEll*pow(10,ellDecimals+2)),min(int((h - a0)*(Battery_Cap*pow(10,Battery_Cap_Fixer+1))),int(round(Battery_Cap*NewEll,hDecimals)*pow(10,Battery_Cap_Fixer+1))),1][0]

					
								
					#Income[min(TempIncome.iteritems(), key=operator.itemgetter(1))[0]] = min(TempIncome.iteritems(), key=operator.itemgetter(1))[1]



			phi = 1
			print("--- %s seconds for phi = 0---" % (time.time() - start_time))

		if phi == 1:
			print "Phi-- 1"
			for i in ellrange:
				ell = i/pow(10,ellDecimals)/100
				hrange = range(0,int(round(Battery_Cap*ell,hDecimals)*pow(10,Battery_Cap_Fixer+1))+int(Battery_Cap_Steps/2),Battery_Cap_Steps)



				for j in hrange:
					h = min(j/(Battery_Cap*ell*pow(10,Battery_Cap_Fixer+1)),1)
					a1_Chosen = max(1 - h,0)
					NewEll =  math.floor(Degradation_Fucntion(ell,h,tau,0,a1_Chosen)*pow(10,ellDecimals+2))/pow(10,ellDecimals+2)
					#print j,h,a1_Chosen


					if ell <= ell_Min_Cap/100 or NewEll <= ell_Min_Cap/100:
						Income[t,i,j,1] = [New_Battery_Cost - Salvage_Value(ell,h) + lamda * Income[t,int(100*pow(10,ellDecimals)),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),0][0],"Replace",int(100*pow(10,ellDecimals)),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),t,"-",'-']
						


					else:
						#for kk in pl.frange(0,math.floor((ii-jj)*100)/100,a1_Step_Size):
						#A1s[t,i,j,1] = a1_Chosen 
						#a1 = GRB.quicksum (a1_Action_Set(a1_Chosen,t)[i] * a1_Probability_set(a1_Action_Set(a1_Chosen,t))[i] for i in range(len(a1_Action_Set(a1_Chosen,t)))).getValue()
						#a1 = round(a1,Decimals)
						a1 = [max(min(xx* PV_Power_Output/(Battery_Cap*ell),1 - h),0.0) for xx in Maximum_Sunlight(t)] #a1_Action_Set(a1_Chosen,t)
								#a1= round(kk,Decimals)
								#NewEll = round(Degradation_Fucntion(ell,h,tau,0,a1),ellDecimals)
						Probs = [1/len(a1)]*len(a1)#[round(1/len(a1),2)]*len() #a1_Probability_set(a1)


								
						NewSElls = list(map(lambda x: math.floor(Degradation_Fucntion(ell,h,tau,0,x)*pow(10,ellDecimals+2))/pow(10,ellDecimals+2), a1))

						#print a1_Chosen,a1,NewSElls
						#print '==================================='
						#print i,j,h,a1,NewSElls,int(round(Battery_Cap*ell,hDecimals)*pow(10,Battery_Cap_Fixer+1)),int(round(Battery_Cap*NewSElls[0],hDecimals)*pow(10,Battery_Cap_Fixer+1))
						#print int((h + a1[0])*(Battery_Cap*NewSElls[0]*pow(10,Battery_Cap_Fixer+1)))-int((h + a1[0])*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)))%Battery_Cap_Steps,int((h + a1[0])*(Battery_Cap*NewSElls[0]*pow(10,Battery_Cap_Fixer+1))),int((h + a1[0])*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)))


						#First = lamda * GRB.quicksum(Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),int((h + a1[x])*(Battery_Cap*NewSElls[x]*pow(10,Battery_Cap_Fixer+1)))-int((h + a1[x])*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)))%Battery_Cap_Steps,0][0]*Probs[x] for x in range(len(a1))).getValue()
						hh = []
						for w,z in enumerate(a1):
							hh.append(takeClosest(hrange, z*(Battery_Cap*ell*pow(10,Battery_Cap_Fixer+1))))
						hrange2 = []
						for w in NewSElls:
							hrange2.append(range(0,int(round(Battery_Cap*w,hDecimals)*pow(10,Battery_Cap_Fixer+1))+int(Battery_Cap_Steps/2),Battery_Cap_Steps)[-1])
						#print ell,h,j,hh,hh[0]+j,hrange[-1],a1,hrange

						




						First = lamda * GRB.quicksum(Income[t,int(round(NewSElls[x]*pow(10,ellDecimals+2))),min(j+hh[x],hrange2[x]),0][0]*Probs[x] for x in range(len(a1))).getValue()
						#if h == 0 and t < Tmax-3:
						#for x in range(len(a1)):
						#	print  t,int(NewSElls[x]*pow(10,ellDecimals+2)),min(j+hh[x],hrange2[x]),hh[x],hrange2[x],Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),min(j+hh[x],hrange2[x]),0]							

						#	print ell,j,NewEll,j+hh[0],a1,Income[t,int(NewSElls[0]*pow(10,ellDecimals+2)),min(j+hh[0],hrange2[0]),0],hrange


						# for x in range(len(a1)):
						# 	print Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),hh[x],0][0]
						# print '====',GRB.quicksum(Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),hh[x],0][0]*Probs[x] for x in range(len(a1))).getValue()

						#min(int((h - a0)*(Battery_Cap*pow(10,Battery_Cap_Fixer+1))),int(round(Battery_Cap*NewSElls[x],hDecimals)*pow(10,Battery_Cap_Fixer+1)))

						Second = New_Battery_Cost - Salvage_Value(ell,h) + lamda * Income[t,100*pow(10,ellDecimals),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),0][0]
						#print First,Second,ell,h,a1[0],h+a1[0],Income[t,int(NewSElls[0]*pow(10,ellDecimals+2)),min(j+hh[0],hrange2[0]),0]


						#print First,Second, lamda * Income[t,100*pow(10,ellDecimals),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),0][0],i,j

						#print '====',i, Income[t,100*pow(10,ellDecimals),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),0][0],GRB.quicksum(Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),hh[x],0][0]*Probs[x] for x in range(len(a1))).getValue(),hh,z*(Battery_Cap*ell*pow(10,Battery_Cap_Fixer+1))

						#TempIncome[t,i,j,1] = [First,"Keep",int(NewEll*pow(10,ellDecimals+2)),(h+a1_Chosen)*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),t,a0*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),t,a1_Chosen*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),Second]
						TempIncome[t,i,j,1] = [First,"Keep",int(NewEll*pow(10,ellDecimals+2)),(h+a1_Chosen)*(Battery_Cap*NewEll*pow(10,Battery_Cap_Fixer+1)),t,a1_Chosen*(Battery_Cap*pow(10,Battery_Cap_Fixer+1)),Second]
						#print ell,h,First,Second


						if First >= Second:
							#print "Hoonga",i
							# if h == 0:
							# 	print t,ell,h,First,Second,NewEll,GRB.quicksum(Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),min(j+hh[x],hrange2[x]),0][0]*Probs[x] for x in range(len(a1))).getValue(),Income[t,100*pow(10,ellDecimals),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),0][0]
							# 	print '=='
							# 	print Income[t,int(NewSElls[0]*pow(10,ellDecimals+2)),min(j+hh[0],hrange2[0]),0],lamda * GRB.quicksum(Income[t,int(NewSElls[x]*pow(10,ellDecimals+2)),min(j+hh[x],hrange2[x]),0][0]*Probs[x] for x in range(len(a1))).getValue()
							# 	print '&&&',ell,Degradation_Fucntion(NewEll,h,tau,0,1)
								# print ell_Min_Cap*pow(10,ellDecimals+2),int(ell*pow(10,ellDecimals+2))
								# for sad in range(ell_Min_Cap*pow(10,ellDecimals),int(ell*pow(10,ellDecimals+2))):
								# 	print t,sad,Income[t,sad,0,1]

						  	TempIncome[t,i,j,1] = [Second,"Replace",100*pow(10,ellDecimals),int(Battery_Cap*pow(10,Battery_Cap_Fixer+1)*.30),t,'-',First]


						
							 		
								 		
							#  	else:
							#  	 	TempIncome[t,ell,h,1] = [Second,"Replace",ell,h,t,a1_Chosen]

							#Income[min(TempIncome.iteritems(), key=operator.itemgetter(1))[0]] = min(TempIncome.iteritems(), key=operator.itemgetter(1))[1]
						Income[t,i,j,1] = TempIncome[t,i,j,1]
						# if h == 0: and t == 360 or h == 0 and t == 361  :
						# 	print t,i,j,Income[t,i,j,1]
						# 	#print("--- %s seconds for phi = 1---" % (time.time() - start_time))

					
						





	
			phi = 0
			print("--- %s seconds for phi = 1---" % (time.time() - start_time))

			if tcurrent - t >= 2 or t == 0:
				Income2 = sorted(Income.items())
				#print Income2[0]

				with open("20Days-4-2-"+str(ell_Min_Cap)+str(t)+"to"+str(tcurrent)+".csv", 'w') as csvfile:
				 	print 'Printing'
				 	spamwriter = csv.writer(csvfile, delimiter=',',
				  	                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
				 	for i in range(len(Income2)):
				 		#print Income[i][0][0],Income[i][0][1],Income[i][0][2],Income[i][0][3],":::::::",Income[i][1][0],Income[i][1][1],Income[i][1][2],Income[i][1][3],Income[i][1][4],Income[i][1][5]
				 		spamwriter.writerow([Income2[i][0][0],Income2[i][0][1],Income2[i][0][2],Income2[i][0][3],":::::::",Income2[i][1][0],Income2[i][1][1],Income2[i][1][2],Income2[i][1][3],Income2[i][1][4],Income2[i][1][5],Income2[i][1][6]])

				print 'Done'
				print("At %s seconds We start cleaning" % (time.time() - start_time))

				Incomers=[]
				for i in Income:
				    Incomers.append(i)
				for i in Incomers:
				    if i[0]!= t:
				        del Income[i]

				Income2 = Incomers = []
				TempIncome={}

				

				print("At %s seconds We Cleaned" % (time.time() - start_time))

				print "Income"+str(t)+"to"+str(tcurrent)+".csv",'Done'
				tcurrent = t






