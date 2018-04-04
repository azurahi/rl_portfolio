import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxopt as opt
from cvxopt import blas, solvers

from market_env import MarketEnv

from keras.optimizers import SGD, Adam

import hyperparameter

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


class DDPG:
	def __init__(self, env, discount = 0.99, model_filename = None, history_filename = None):
		self.env = env
		self.discount = discount
		self.model_filename = model_filename
		self.history_filename = history_filename



		self.model = DDPG(modelFilename).getModel()
		sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
		adam = Adam(lr = 0.001, decay=1e-6)
		self.model.compile(loss='mse', optimizer=adam)

	def discount_rewards(self, r):
		discounted_r = np.zeros_like(r)
		running_add = 0
		r = r.flatten()

		for t in reversed(range(0, r.size)):
			if r[t] != 0:
				running_add = 0

			running_add = running_add * self.discount + r[t]
			discounted_r[t] = running_add
		return discounted_r

	def train(self, max_episode = 300, max_path_length = 200, verbose = 1):
		env = self.env
		model = self.model
		avg_reward_sum = 0.

		for e in range(max_episode):
			env.reset()
			observation = env.reset()
			game_over = False
			reward_sum = 1

			inputs = []
			outputs = []
			predicteds = []
			rewards = []

			while not game_over:
				aprob = model.predict(observation)[0]
				inputs.append(observation)
				predicteds.append(aprob)
				
				action = aprob
				outputs.append(action)

				# self.env.portfolio_weight = action
				# print(action)
				observation, reward, game_over, info, sharpe, future_returns = self.env.step(action)
				if reward != 0:
					reward_sum *= float(1.+reward)

				rewards.append(float(reward))
				
			avg_reward_sum = avg_reward_sum * 0.99 + reward_sum * 0.01
			toPrint = "%d\t%s\t%s\t%.2f\t%.2f\t%.2f" % (e, "포트폴리오", "%.2f" %reward_sum, info["cum"], info["equally"], info["ret"])
			print(toPrint)
			toPrint = "%d\t%s\t%s\t%.2f\t%.2f\t%.2f" % (e, "포트폴리오","%.2f" %reward_sum, sharpe["benchmark"], sharpe["equally"], sharpe["reward"])
			print(toPrint)
			if self.history_filename != None:
				os.system("echo %s >> %s" % (toPrint, self.history_filename))


			dim = len(inputs[0])
			inputs_ = [[] for i in range(dim)]
			for obs in inputs:
				for i, block in enumerate(obs):
					inputs_[i].append(block[0])
			inputs_ = [np.array(inputs_[i]) for i in range(dim)]
			outputs_ = np.vstack(outputs)
			predicteds_ = np.vstack(predicteds)
			rewards_ = np.vstack(rewards)
			# print(rewards_)
			discounted_rewards_ = self.discount_rewards(rewards_)
			#discounted_rewards_ -= np.mean(discounted_rewards_)
			discounted_rewards_ /= np.std(discounted_rewards_)

			# outputs_ *= discounted_rewards_
			for i, r in enumerate(zip(rewards, discounted_rewards_)):
				reward, discounted_reward = r
				# print(r)
				# if verbose > 1:
				# 	print(outputs_[i],)
				# print(outputs_[i])
				# outputs_[i] = 0.5 + (2 * outputs_[i] - 1) * discounted_reward_
				if discounted_reward < 0:
					opt_port = optimal_portfolio(future_returns[i])
					outputs_[i] = opt_port.T
					# outputs_[i] = outputs_[i] / sum(outputs_[i])
				# outputs_[i] = np.minimum(1, np.maximum(0, predicteds_[i] + (outputs_[i] - predicteds_[i]) * abs(discounted_reward)))
				# print(outputs_)
				if verbose > 1:
					print(predicteds_[i], outputs_[i])
			outputs_[i] = outputs_[i] / sum(outputs_[i])
			# data_tracker(inputs, outputs, predicteds)
			model.fit(inputs_, outputs_, epochs = 1, verbose = 1, shuffle = False)
			model.save_weights(self.model_filename)

	def test(self, max_episode=1, max_path_length=200, verbose=1):
		env = self.env
		model = self.model
		avg_reward_sum = 0.

		for e in range(max_episode):
			env.reset()
			observation = env.reset()
			game_over = False
			reward_sum = 1.
			markowitz_sum = 1.
			equally_sum = 1.

			inputs = []
			outputs = []
			predicteds = []
			rewards = []

			reward_sum_list = []
			equally_sum_list = []
			markowitz_sum_list = []

			while not game_over:
				aprob = model.predict(observation)[0]
				inputs.append(observation)
				predicteds.append(aprob)

				action = aprob
				outputs.append(action)

				# self.env.portfolio_weight = action
				# print(action)
				observation, reward, game_over, info, sharpe, future_returns = self.env.step(action)
				if reward != 0:
					reward_sum *= float(1. + reward)
					reward_sum_list.append(reward_sum)
				if info["cum"] != 0:
					markowitz_sum_list.append(info["cum"])
				if info["equally"] != 0:
					equally_sum_list.append(info["equally"])
				rewards.append(float(reward))

			avg_reward_sum = avg_reward_sum * 0.99 + reward_sum * 0.01
			toPrint = "%d\t%s\t%s\t%.2f\t%.2f\t%.2f" % (
			e, "포트폴리오", "%.2f" %reward_sum,
			info["cum"], info["equally"], info["ret"])
			print(toPrint)
			toPrint = "%d\t%s\t%s\t%.2f\t%.2f\t%.2f" % (
			e, "포트폴리오","%.2f" %reward_sum,
			sharpe["benchmark"], sharpe["equally"], sharpe["reward"])
			print(toPrint)
			if self.history_filename != None:
				os.system("echo %s >> %s" % (toPrint, self.history_filename))
			graphy(reward_sum_list, markowitz_sum_list, equally_sum_list)

def graphy(data_list1,data_list2,data_list3):
	plt.plot(data_list1)
	plt.plot(data_list2)
	plt.plot(data_list3)
	plt.show()

def data_tracker(inputs, outputs, weight):
	data = pd.DataFrame(data = {'inputs':inputs, 'outputs': outputs, 'weights':weight})
	data.to_csv('data.csv')

if __name__ == "__main__":
	import sys
	import codecs

	codelist_filename = sys.argv[1]
	envList_filename = sys.argv[2]
	model_filename = sys.argv[3] if len(sys.argv) > 3 else None
	history_filename = sys.argv[4] if len(sys.argv) > 4 else None

	codeMap = {}
	envMap = {}
	f = codecs.open(codelist_filename, "r", encoding = "euc-kr", errors='ignore')

	for line in f:
		if line.strip() != "":
			tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
			codeMap[tokens[0]] = tokens[1]
	print(list(codeMap.keys()))

	f.close()

	ff = codecs.open(envList_filename, "r", encoding="euc-kr", errors='ignore')

	for line in ff:
		if line.strip() != "":
			tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
			envMap[tokens[0]] = tokens[1]
	print(list(envMap.keys()))

	ff.close()

	env = MarketEnv(dir_path = "./data/", target_codes = list(codeMap.keys()), input_codes = list(envMap.keys()), start_date = "2010-08-01", end_date = "2016-12-30", sudden_death = -1.0)

	pg = DDPG(env, discount = 0.9, model_filename = model_filename, history_filename = history_filename)
	pg.train(verbose = 1)

	env = MarketEnv(dir_path = "./data/", target_codes = list(codeMap.keys()), input_codes = list(envMap.keys()), start_date = "2016-06-03", end_date = "2016-12-30", sudden_death = -1.0)

	pg = DDPG(env, discount = 0.9, model_filename = model_filename, history_filename = history_filename)
	pg.test(verbose = 1)

