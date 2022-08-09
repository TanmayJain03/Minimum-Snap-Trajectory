# library imports 
from math import factorial
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import numpy as np
from scipy.linalg import block_diag

class MinSnap:
	def __init__(self):
		# constant variables
		self.V_CONST = 2.0
		self.POLYNOMIAL_ORDER = 7	

		self.time_stamps = [0]

		# constants upto 7 derivatives 
		self.set_derivative()

		self.n, self.waypoints = self.inputPoints()
		# no. of polynomials = no. of points - 1
		self.m = self.n - 1
		self.q = np.zeros((8*self.m, 1))
		self.G = np.zeros((4*self.m + 2, 8*self.m))
		self.h = np.zeros((4*self.m + 2, 1)).reshape((4*self.m+2,))

		# create 2 matrices for each axes to solve AX = B where A: 8m*8m matrix and B: 1*8m matrix

		self.Ax = np.zeros((4*self.m+2, 8*self.m))
		self.Ay = np.zeros((4*self.m+2, 8*self.m))
		self.Az = np.zeros((4*self.m+2, 8*self.m))
		self.Bx = np.zeros((4*self.m+2, 1))
		self.By = np.zeros((4*self.m+2, 1))
		self.Bz = np.zeros((4*self.m+2, 1))
	
		self.A = [self.Ax, self.Ay, self.Az]
		self.B = [self.Bx, self.By, self.Bz]

		#length between waypoints
		self.len_bw_waypoints = self.lenBwPoints()

		#time taken to travel from ith waypoint to (i+1)th waypoint
		self.time_bw_waypoints = self.timeBwPoints(self.len_bw_waypoints)

		# creating time array for polynomial of 8 terms wrt T for each ti
		self.time_array = self.timeArray(self.time_bw_waypoints)
		

		self.X = []
		self.Y = []
		self.Z = []

	# functions

	def set_derivative(self):
		self.d = [[0]]*8
		self.d[0] = [1, 1, 1, 1, 1, 1, 1, 1]
		self.d[1] = [0, 1, 2, 3, 4, 5, 6, 7]
		self.d[2] = [0, 0, 2, 6, 12, 20, 30, 42]
		self.d[3] = [0, 0, 0, 6, 24, 60, 120, 210]
		self.d[4] = [0, 0, 0, 0, 24, 120, 360, 840]
		self.d[5] = [0, 0, 0, 0, 0, 120, 720, 2520]
		self.d[6] = [0, 0, 0, 0, 0, 0, 720, 5040]
		self.d[7] = [0, 0, 0, 0, 0, 0, 0, 5040]

	def inputPoints(self):
		n = int(input("Enter number of points in 3D space\n"))
		print(f"Enter the x, y, z coordinates for the "+str(n)+" points")
		vector = []
		for i in range(n):
			x, y, z = [int(x) for x in input(f"Enter x, y, z for point {i+1} :").split()]
			vector.append([x, y, z])
		self.V_CONST = int(input("Enter the uniform speed of the point\n"))
		return n, vector

	def lenBwPoints(self):
		len = np.zeros((self.m))
		for i in range(0,self.m):
			len[i] = np.linalg.norm(np.array(self.waypoints[i+1]) - np.array(self.waypoints[i]))
		return len

	def timeBwPoints(self, len_bw_waypoints):
		time = np.zeros((self.m))
		cumulative_time = 0
		for i in range(self.m):
			time[i] = len_bw_waypoints[i]/self.V_CONST
			cumulative_time += time[i]
			self.time_stamps.append(cumulative_time)
			time[i] = cumulative_time
		return time

	def timeArray(self, time_bw_waypoints):
		timeA = np.zeros((8*self.m, 1))
		for i in range(self.m):
			for j in range(0,8):
				timeA[i*8 + j][0] = time_bw_waypoints[i]**j
		return timeA

	def firstEq(self, position_variable):                  #position_variable 0=x, 1=y, 2=z
		self.A[position_variable][0][0] = 1
		self.B[position_variable][0][0] = self.waypoints[0][position_variable]

	def equatePointsAndPolynomial(self, position_variable):  #m+1 pi(ti) = point(i)
		for row in range(1,self.m+1):
			for i in range((row-1)*8, row*8):
				self.A[position_variable][row][i] = self.d[0][i%8]*self.time_array[i][0]
			self.B[position_variable][row][0] = self.waypoints[row][position_variable]

	def continousDerivativeEquations(self, position_variable):  # 3(m-1)     der [0->2]    // m+1 + 3(m-1) = 4m - 2
		for der in range(0,3):
			for row in range(self.n + der*(self.m-1), self.n + (der+1)*(self.m-1)):
				for i in range((row - (self.n + der*(self.m-1)))*8, (row - (self.n + der*(self.m-1)))*8 + 8):
					self.A[position_variable][row][i] = 1*self.d[der][i%8]*self.time_array[i][0]
					self.A[position_variable][row][i+8] = -1*self.d[der][i%8]*self.time_array[i][0]

	def endpointEquations(self, position_variable):  # 4 der [1,2]
		for der in range(1,3):
			for row in range(4*self.m-4 + 2*der, 4*self.m-4 + 2*(der+1), 2):
				for i in range(0,8):
					if i==der:
						self.A[position_variable][row][i] = self.d[der][i]
					self.A[position_variable][row+1][8*self.m-8+i] = self.d[der][i]*self.time_array[8*self.m-8+i]

	def makeQMatrixElement(self, i):
		Qi = np.zeros((self.POLYNOMIAL_ORDER+1, self.POLYNOMIAL_ORDER+1))
		for row in range (0, self.POLYNOMIAL_ORDER + 1):
			for col in range (0, self.POLYNOMIAL_ORDER + 1):
				if(row < 4 or col < 4):
					continue
				Qi[row][col] = (factorial(row)/factorial(row-4))*(factorial(col)/factorial(col-4))*(1/((row-4) + (col-4) + 1))*(self.time_stamps[i]**(row+col-7) - (self.time_stamps[i-1]**(row+col-7)))
		return Qi

	#block diagonals
	def makeQMatrixFull(self):
		QList = []
		Q = np.zeros((self.m, self.m))
		for row in range(0, self.m):
			for col in range(0, self.m):
				if row == col:
					QList.append(self.makeQMatrixElement(row+1))
					#Q[row][col] = makeQMatrixElement(row+1)
			
		Q = []
		tmp = QList[0]
		for i in range(0, len(QList)-1):
			Q = block_diag(tmp, QList[i+1])  #+ (0.0001 * np.identity((8*m)))
			tmp = Q
		Q += (0.0001 * np.identity((8*self.m)))
		return Q

	def generateMinSnapPoly(self, position_variable):
		##### FIRST 2m constraints
		# first equation
		self.firstEq(position_variable)

		# next m equations equating postion at time of reaching waypoint for the previous polynomial
		self.equatePointsAndPolynomial(position_variable)

		# next m-1 equations equating position at time of reachin waypoint for the next polynomial
		#previousPointsAndPolynomials(position_variable)

		#####  NEXT 6m-6 constraints
		self.continousDerivativeEquations(position_variable)

		##### LAST 6 constraints
		self.endpointEquations(position_variable)	

	def solveEquations(self):
		###### FOR X, Y, Z AXES

		self.generateMinSnapPoly(0)
		self.generateMinSnapPoly(1)
		self.generateMinSnapPoly(2)
		self.Ax, self.Ay, self.Az = self.A[0], self.A[1], self.A[2]
		self.Bx, self.By, self.Bz = self.B[0].T, self.B[1].T, self.B[2].T

		QMatrix = self.makeQMatrixFull()      

		##### SOLVE MATRIX EQUATIONS

		self.X = solve_qp(QMatrix , self.q, self.G, self.h, self.Ax, self.Bx[0], solver='osqp')
		self.Y = solve_qp(QMatrix , self.q, self.G, self.h, self.Ay, self.By[0], solver='osqp')
		self.Z = solve_qp(QMatrix , self.q, self.G, self.h, self.Az, self.Bz[0], solver='osqp')
		
	def plotPoly(self):	
		fig = plt.figure()
		ax = plt.axes(projection="3d")
		ax.scatter([row[0] for row in self.waypoints], [row[1] for row in self.waypoints], [row[2] for row in self.waypoints], marker = 'o')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		for i in range(0, self.m):
			t = np.linspace(self.time_stamps[i], self.time_stamps[i + 1], 100)

			x = []
			y = []
			z = []
			for k in range(0,100):
				x_, y_, z_ = 0, 0, 0
				for j in range(i*8, (i+1)*8):
					x_ += self.X[j]*t[k]**(j%8)
					y_ += self.Y[j]*t[k]**(j%8)
					z_ += self.Z[j]*t[k]**(j%8)
				x.append(x_)
				y.append(y_)
				z.append(z_)
			ax.plot3D(x,y,z, 'b')
		plt.show()

	

	## MAIN PROGRAM
if __name__ == '__main__':
	Min_snap = MinSnap()
	Min_snap.solveEquations()
	Min_snap.plotPoly()
