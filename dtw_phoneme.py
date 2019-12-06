########################################################################
#                                                                      #
#   Sebastião Quintas - @IRIT SAMoVA - 06/12/2019
#                                                                      #
########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import csv

def distance_cost_plot(distances,target,predicted,x_path,y_path):
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds')    
    plt.gca().invert_yaxis()
    tick_marks_x = np.arange(len(target))
    tick_marks_y = np.arange(len(predicted))
    plt.xticks(tick_marks_y,predicted)
    plt.yticks(tick_marks_x,target)  
    plt.xlabel('Predicted')
    plt.ylabel('Target')   
    plt.colorbar()
    plt.plot(x_path,y_path)
    plt.savefig('dtw_align.png')
    plt.show()
    plt.clf()

def read_from_file(file1,file2):	
	#Transform input files from kaldi format to phoneme array.
	
	targ = []
	with open(file1,"r") as fd:
		lines = fd.read().splitlines()
		for line in lines:
			fields = line.split(" ")
			targ.append(fields[3])	
	
	pred = []
	with open(file2,"r") as fd:
		lines = fd.read().splitlines()
		for line in lines:
			fields = line.split(" ")
			pred.append(fields[3])	
	
	return targ,pred
	
def ret_ind(char,file3):
	with open(file3,'r') as fd:
		lines = fd.read().splitlines()
		for line in lines:
			fields = line.split(" ")
			if (char == fields[0]):
				return int(fields[1])
				
	print("Character not recnognized!")
	sys.exit()

def backtracking(x,y,accum_cost):
	path = [[len(x)-1, len(y)-1]]
	i = len(x)-1
	j = len(y)-1
	dist = 0
	while i>0 and j>0:
		if i==0:
			j=j-1
		elif j==0:
			i=i-1			
		else:
			if accum_cost[i-1,j]==min(accum_cost[i-1,j-1], accum_cost[i-1,j], accum_cost[i,j-1]):
				i=i-1
			elif accum_cost[i,j-1]==min(accum_cost[i-1,j-1], accum_cost[i-1,j], accum_cost[i,j-1]):
				j=j-1
			else:
				j=j-1
				i=i-1
		path.append([i,j])
	#path.append([0,0])
	return path
		
def accum_cost(x,y,distance_matrix):

	accum_cost = np.zeros((len(x),len(y)))
	accum_cost[0,0] = distance_matrix[0,0]

	for i in range(1, len(y)):
		accum_cost[0,i] = distance_matrix[0,i]+accum_cost[0,i-1]

	for i in range(1, len(x)):
		accum_cost[i,0] = distance_matrix[i,0]+accum_cost[i-1,0]
	
	for i in range(1, len(x)):
		for j in range(1, len(y)):
			accum_cost[i,j] = distance_matrix[i,j]+min(accum_cost[i-1,j-1], accum_cost[i-1,j], accum_cost[i,j-1])
			
	return accum_cost

	
	
if __name__=="__main__":
	
	if len(sys.argv)<4:
		print('python '+sys.argv[0]+' <target.txt> <predicted.txt> <phonemes.txt> <dist.csv>')
		sys.exit()
	
	target, predicted = read_from_file(sys.argv[1],sys.argv[2])
						 
	# Load phoneme-distance matrix
	with open(sys.argv[4], newline='') as csvfile:
		total_dist = list(csv.reader(csvfile))
	total_dist = [list(map(int,x)) for x in total_dist]
	total_dist = np.asarray(total_dist)

	# Compute the distance matrix between target and predicted sequences
	dist_mat = np.zeros((len(target),len(predicted)))
	for i in range(len(target)):
		for j in range(len(predicted)):
			dist_mat[i,j] = total_dist[ret_ind(target[i],sys.argv[3]),ret_ind(predicted[j],sys.argv[3])]**2
	
	# Compute the accumulated cost of all paths
	accumulated_cost = accum_cost(target,predicted,dist_mat)
	
	# Backtracks the accumulated cost matrix in a greedy-search way
	path = backtracking(target,predicted,accumulated_cost)
	path_pred = [point[0] for point in path]
	path_targ = [point[1] for point in path]
	
	distance_cost_plot(accumulated_cost,target,predicted,path_targ,path_pred)
	
	total_distance = 0
	for i in range(len(path)):
		total_distance += accumulated_cost[path_pred[i],path_targ[i]]
	
	# Similarity score computation
	similarity_score = 1/(1+np.sqrt(total_distance))
	print('Similarity score between the two sequences: {0:.3f}'.format(similarity_score))
	

