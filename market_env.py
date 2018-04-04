from random import random
import numpy as np
import math
import pandas as pd
import cvxopt as opt
from cvxopt import solvers
import gym
from gym import spaces

import hyperparameter

def sharpe_ratio(returns,):	return np.mean(returns)/np.std(returns)

def optimal_portfolio(returns,):

	solvers.options['show_progress'] = False

	n = len(returns)
	returns = np.asmatrix(returns)

	pbar = opt.matrix(np.mean(returns, axis=1))

	r_mean = np.mean(returns)
	P = opt.matrix(np.cov(returns))
	q = opt.matrix(np.zeros((n, 1)), tc='d')
	G = -opt.matrix(np.concatenate((
		np.array(pbar.T),
		np.eye(n)), 0))
	h = opt.matrix(np.concatenate((
		-np.ones((1, 1)) * r_mean,
		np.zeros((n, 1))), 0))

	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)
	sol = solvers.qp(P, q, G, h, A, b)['x']
	return np.asarray(sol)

class MarketEnv(gym.Env):
	PENALTY = 1  # 0.999756079

	def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope=20, sudden_death=-1., cumulative_reward=False):
		self.startDate = start_date
		self.endDate = end_date
		self.scope = scope
		self.sudden_death = sudden_death
		self.cumulative_reward = cumulative_reward
		self.futureReturn = []
		self.inputCodes = []
		self.targetCodes = []
		self.weightReward = []
		self.dataMap = {}

		self.reward_list = []
		self.benchmark_list = []
		self.equally_list = []

		# self.riskfree = -0.0003
		print(target_codes)
		print(input_codes)
		for code in list(set(target_codes + input_codes)):
			fn = dir_path + code + ".csv"
			data = {}
			lastClose = 0
			lastVolume = 0
			try:
				f = open(fn, "r",encoding= 'euc-kr')
				next(f)
				for line in f:
					if line.strip() != "":
						dt, openPrice, close, volume, per, volume_ind, volume_comp, volume_fore = line.strip().split(",")
						try:
							if dt >= self.startDate:
								# high = float(high) if high != "" else float(close)
								# low = float(low) if low != "" else float(close)
								openPrice = float(openPrice)
								close = float(close)
								per = 0.0 if per == "" else float(per)
								volume_ind = int(volume_ind)
								volume_comp = int(volume_comp)
								volume_fore = int(volume_fore)
								volume = float(volume)

								if lastClose > 0 and close > 0 and lastVolume > 0:
									openPrice_ = (openPrice -lastClose) / lastClose if lastClose != 0 else 0.0001
									close_ = (close - lastClose) / lastClose if lastClose != 0 else 0.0001
									volume_ = (volume - lastVolume) / lastVolume if lastVolume != 0 else 0.0001
									per_ = per
									volume_ind_ = (volume_ind - volume)/volume if lastVolume != 0 else 0.0001
									volume_comp_ = (volume_comp - volume)/volume if lastVolume != 0 else 0.0001
									volume_fore = (volume_fore -volume)/volume if lastVolume != 0 else 0.0001

									data[dt] = (openPrice_, close_, volume_,per_,volume_ind_,volume_comp_, volume_fore)

								lastClose = close
								lastVolume = volume
						except Exception as e:
							print(code)
							print(e, line.strip().split(","))

						# try:
						# 	if dt
				f.close()
			except Exception as e:
				print(e)

			if len(data.keys()) > scope:
				self.dataMap[code] = data
				data_d = pd.DataFrame(data)
				data_d.to_csv('test.csv')
				if code in target_codes:
					self.targetCodes.append(code)
					self.targetCodes.sort()
				if code in input_codes:
					self.inputCodes.append(code)
					self.inputCodes.sort()
		self.portfolio_weight = np.ones(len(self.targetCodes)) / (len(self.targetCodes))
		self.action_space = spaces.Box(np.zeros((len(target_codes))) , np.ones((len(target_codes))))
		self.observation_space = spaces.Box(np.ones(scope * len(input_codes)) * -1,
											np.ones(scope * len(input_codes)))

		self.reset()
		self._seed()

	def _step(self, action):
		if self.done:
			return self.state, self.reward, self.done, {}
		self.portfolio_weight = action
		self.weightReward = []
		self.futureReturn = []
		self.pastReturn = []
		
		try:
			for i in self.targetCodes:
				self.weightReward.append(self.dataMap[i][self.targetDates[self.currentTargetIndex]][1])
		except Exception as e:
			print(e)
		try:
			for j in range(self.currentTargetIndex):
				p_list = []
				for i in self.targetCodes:
					d_list = []
					for date in range(j,j+self.scope):
						d_list.append(self.dataMap[i][self.targetDates[date]][1])
					p_list.append(d_list)
				self.futureReturn.append(p_list)
		except Exception as e:
			print(e)
		try:
			for i in self.targetCodes:
				d_list = []
				for date in range(self.currentTargetIndex-self.scope,self.currentTargetIndex):
					d_list.append(self.dataMap[i][self.targetDates[date]][1])
				self.pastReturn.append(d_list)
		except Exception as e:
			print(e)
		self.weightReward = np.array(self.weightReward)

		self.reward = 0
		self.reward = sum(action * self.weightReward)
		self.benchmark = sum(optimal_portfolio(self.pastReturn)[0] * self.weightReward)
		self.ret = self.ret * (1 + self.reward)
		self.cum = self.cum * (1 + self.benchmark)
		equally = sum(self.weightReward)/len(self.weightReward)
		self.equally = self.equally * (1 + equally)
		self.reward_list.append(self.reward)
		self.benchmark_list.append(self.benchmark)
		self.equally_list.append(equally)



		self.defineState()
		self.currentTargetIndex += 1
		if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
			self.currentTargetIndex]:
			self.done = True

		return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum, "equally": self.equally,"ret":self.ret },{"reward": sharpe_ratio(self.reward_list),"equally":sharpe_ratio(self.equally_list),"benchmark":sharpe_ratio(self.benchmark_list)}, self.futureReturn

	def _reset(self):
		self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
		self.target = self.dataMap[self.targetCode]
		self.inputs = {}
		for i in self.inputCodes:
			self.inputs[i] = self.dataMap[i]

		self.targetDates = sorted(self.target.keys())
		self.currentTargetIndex = self.scope
		self.cum = 1.
		self.equally = 1.
		self.ret = 1.
		self.done = False
		self.reward = 0

		self.reward_list = []
		self.benchmark_list = []
		self.equally_list = []

		self.defineState()


		return self.state

	def _render(self, mode='human', close=False):
		if close:
			return
		return self.state

	'''
	def _close(self):
		pass

	def _configure(self):
		pass
	'''

	def _seed(self):
		return int(random() * 100)

	def defineState(self):
		tmpState = []
		tmpState.append([self.portfolio_weight])

		subject = []
		subject1 = []
		subject2 = []
		subject3 = []  
		subject4 = []
		subject5 = []
		subjectVolume = []
		for i in range(self.scope):
			try:
				for inp in self.inputCodes:
					subject1.append([self.inputs[inp][self.targetDates[self.currentTargetIndex - 1 - i]][0]])
					subject2.append([self.inputs[inp][self.targetDates[self.currentTargetIndex - 1 - i]][2]])
					subject3.append([self.inputs[inp][self.targetDates[self.currentTargetIndex - 1 - i]][4]])
					subject4.append([self.inputs[inp][self.targetDates[self.currentTargetIndex - 1 - i]][5]])
					subjectVolume.append([self.inputs[inp][self.targetDates[self.currentTargetIndex - 1 - i]][6]])
			except Exception as e:
				self.done = True
		tmpState.append([[subject1,subject2, subject3,subject4, subjectVolume]])
		tmpState = [np.array(i) for i in tmpState]
		self.state = tmpState
