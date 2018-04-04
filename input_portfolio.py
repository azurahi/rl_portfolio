# from random import random
# import pandas as pd
# import numpy as np
# import math
#
# import hyperparameter as hp
#
# class DataProcesser:
#     def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope = hp.scope):
#         self.startDate = start_date
#         self.endDate = end_date
#         self.scope = scope
#
#         self.inputCodes = []
#         self.targetCodes = []
#         self.dataMap = {}
#
#         for code in list(set(target_codes + input_codes)):
#             fn = dir_path + code + ".csv"
#             data = {}
#             lastClose = 0
#             lastVolume = 0
#             try:
#                 f = open(fn, "r", encoding='euc-kr')
#                 next(f)
#                 for line in f:
#                     if line.strip() != "":
#                         dt, openPrice, close, volume, per, volume_ind, volume_comp, volume_fore = line.strip().split(",")
#                         try:
#                             if dt >= start_date:
#                                 openPrice = float(openPrice)
#                                 close = float(close)
#                                 per = 0.0 if per == "" else float(per)
#                                 volume_ind = int(volume_ind)
#                                 volume_comp = int(volume_comp)
#                                 volume_fore = int(volume_fore)
#                                 volume = float(volume)
#
#                                 if lastClose > 0 and close > 0 and lastVolume > 0:
#                                     openPrice_ = (openPrice - lastClose) / lastClose
#                                     close_ = (close - lastClose) / lastClose
#                                     # high_ = (high - close) / close
#                                     # low_ = (low - close) / close
#                                     volume_ = (volume - lastVolume) / lastVolume
#                                     per_ = per
#                                     volume_ind_ = (volume_ind - volume) / volume
#                                     volume_comp_ = (volume_comp - volume) / volume
#                                     volume_fore_ = (volume_fore - volume) / volume
#
#                                     data[dt] = (openPrice_, close_, volume_, per_, volume_ind_, volume_comp_, volume_fore_)
#
#                                 lastClose = close
#                                 lastVolume = volume
#                         except Exception as e:
#                             print(e, line.strip().split(","))
#                 f.close()
#             except Exception as e:
#                 print(e)
#
#
#
#     def regulizer(self, openPrice, close, volume, per, volume_ind, volume_comp, volume_fore):
#         openPrice = float(openPrice)
#         close = float(close)
#         per = 0.0 if per == "" else float(per)
#         volume_ind = int(volume_ind)
#         volume_comp = int(volume_comp)
#         volume_fore = int(volume_fore)
#         volume = float(volume)
#
# class MarketEnv(gym.Env):
# 	PENALTY = 1  # 0.999756079
#
# 	def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope=hp.scope, sudden_death=-1.,cumulative_reward=False):
# 		self.startDate = start_date
# 		self.endDate = end_date
# 		self.scope = scope
# 		self.sudden_death = sudden_death
# 		self.cumulative_reward = cumulative_reward
#
# 		self.inputCodes = []
# 		self.targetCodes = []
# 		self.dataMap = {}
# 		print(target_codes)
# 		print(input_codes)
# 		for code in list(set(target_codes + input_codes)):
# 			fn = dir_path + code + ".csv"
# 			data = {}
# 			lastClose = 0
# 			lastVolume = 0
# 			try:
# 				f = open(fn, "r",encoding= 'euc-kr')
# 				next(f)
# 				for line in f:
# 					if line.strip() != "":
# 						dt, openPrice, close, volume, per, volume_ind, volume_comp, volume_fore = line.strip().split(",")
# 						try:
# 							if dt >= start_date:
# 								# high = float(high) if high != "" else float(close)
# 								# low = float(low) if low != "" else float(close)
# 								openPrice = float(openPrice)
# 								close = float(close)
# 								per = 0.0 if per == "" else float(per)
# 								volume_ind = int(volume_ind)
# 								volume_comp = int(volume_comp)
# 								volume_fore = int(volume_fore)
# 								volume = float(volume)
#
# 								if lastClose > 0 and close > 0 and lastVolume > 0:
# 									openPrice_ = (openPrice -lastClose) / lastClose
# 									close_ = (close - lastClose) / lastClose
# 									# high_ = (high - close) / close
# 									# low_ = (low - close) / close
# 									volume_ = (volume - lastVolume) / lastVolume
# 									per_ = per
# 									volume_ind_ = (volume_ind - volume)/volume
# 									volume_comp_ = (volume_comp - volume)/volume
# 									volume_fore = (volume_fore -volume)/volume
#
# 									data[dt] = (openPrice_, close_, volume_,per_,volume_ind_,volume_comp_, volume_fore)
#
# 								lastClose = close
# 								lastVolume = volume
# 						except Exception as e:
# 							print(e, line.strip().split(","))
# 				f.close()
# 			except Exception as e:
# 				print(e)
#
# 			if len(data.keys()) > scope:
# 				self.dataMap[code] = data
# 				data_d = pd.DataFrame(data)
# 				data_d.to_csv('test.csv')
# 				if code in target_codes:
# 					self.targetCodes.append(code)
# 				if code in input_codes:
# 					self.inputCodes.append(code)

self = 0.003
print([self] * 5)