crop_disease = {'柑桔健康': 24,'柑桔黄龙病': 14,
 '桃子健康': 36,
 '桃疮痂病': 16,
 '樱桃健康': 28,
 '樱桃白粉病': 6,
 '玉米健康': 27,
 '玉米叶斑病': 15,
 '玉米灰斑病': 19,
 '玉米花叶病毒病': 26,
 '玉米锈病': 11,
 '番茄健康': 30,
 '番茄叶霉病': 13,
 '番茄斑枯病': 9,
 '番茄斑点病': 18,
 '番茄早疫病': 1,
 '番茄晚疫病菌': 3,
 '番茄疮痂病': 5,
 '番茄白粉病': 0,
 '番茄红蜘蛛损伤': 20,
 '番茄花叶病毒病': 31,
 '番茄黄化曲叶病毒病': 7,
 '苹果健康': 34,
 '苹果灰斑病': 25,
 '苹果雪松锈病': 17,
 '苹果黑星病': 12,
 '草莓健康': 35,
 '草莓叶枯病': 2,
 '葡萄健康': 32,
 '葡萄褐斑病': 8,
 '葡萄轮斑病': 22,
 '葡萄黑腐病': 23,
 '辣椒健康': 33,
 '辣椒疮痂病': 4,
 '马铃薯健康': 29,
 '马铃薯早疫病': 10,
 '马铃薯晚疫病': 21}


crop_degree = {'柑橘一般': 15,
 '柑橘严重': 16,
 '柑橘健康': 14,
 '桃一般': 18,
 '桃严重': 19,
 '桃健康': 17,
 '樱桃一般': 5,
 '樱桃严重': 6,
 '樱桃健康': 4,
 '玉米一般': 8,
 '玉米严重': 9,
 '玉米健康': 7,
 '玉米花叶病毒': 10,
 '番茄一般': 30,
 '番茄严重': 31,
 '番茄健康': 29,
 '番茄花叶病毒': 32,
 '苹果一般': 1,
 '苹果严重': 2,
 '苹果健康': 0,
 '苹果灰斑病': 3,
 '草莓一般': 27,
 '草莓严重': 28,
 '草莓健康': 26,
 '葡萄一般': 12,
 '葡萄严重': 13,
 '葡萄健康': 11,
 '辣椒一般': 21,
 '辣椒严重': 22,
 '辣椒健康': 20,
 '马铃薯严重': 25,
 '马铃薯健康': 23,
'马铃薯一般':24}

crop_disease_label={0: 34,
 1: 12,
 2: 12,
 3: 25,
 4: 17,
 5: 17,
 6: 28,
 7: 6,
 8: 6,
 9: 27,
 10: 19,
 11: 19,
 12: 11,
 13: 11,
 14: 15,
 15: 15,
 16: 26,
 17: 32,
 18: 23,
 19: 23,
 20: 22,
 21: 22,
 22: 8,
 23: 8,
 24: 24,
 25: 14,
 26: 14,
 27: 36,
 28: 16,
 29: 16,
 30: 33,
 31: 4,
 32: 4,
 33: 29,
 34: 10,
 35: 10,
 36: 21,
 37: 21,
 38: 35,
 39: 2,
 40: 2,
 41: 30,
 42: 0,
 43: 0,
 44: 5,
 45: 5,
 46: 1,
 47: 1,
 48: 3,
 49: 3,
 50: 13,
 51: 13,
 52: 18,
 53: 18,
 54: 9,
 55: 9,
 56: 20,
 57: 20,
 58: 7,
 59: 7,
 60: 31}

crop_degree_label = {0: 0,
 1: 1,
 2: 2,
 3: 3,
 4: 1,
 5: 2,
 6: 4,
 7: 5,
 8: 6,
 9: 7,
 10: 8,
 11: 9,
 12: 8,
 13: 9,
 14: 8,
 15: 9,
 16: 10,
 17: 11,
 18: 12,
 19: 13,
 20: 12,
 21: 13,
 22: 12,
 23: 13,
 24: 14,
 25: 15,
 26: 16,
 27: 17,
 28: 18,
 29: 19,
 30: 20,
 31: 21,
 32: 22,
 33: 23,
 34: 24,
 35: 25,
 36: 24,
 37: 25,
 38: 26,
 39: 27,
 40: 28,
 41: 29,
 42: 30,
 43: 31,
 44: 30,
 45: 31,
 46: 30,
 47: 31,
 48: 30,
 49: 31,
 50: 30,
 51: 31,
 52: 30,
 53: 31,
 54: 30,
 55: 31,
 56: 30,
 57: 31,
 58: 30,
 59: 31,
 60: 32}

crop = {'苹果': 0, '樱桃': 1, '玉米': 2, '葡萄': 3, '柑桔': 4, '辣椒': 5, '马铃薯': 6, '草莓': 7, '番茄': 8, '桃子': 9, }

disease = {'健康': 0, '黑星': 1, '灰斑': 2, '锈病': 3, '白粉': 4, '叶斑': 5, '花叶病毒': 6, '黑腐': 7, '轮斑': 8, '褐斑': 9, '黄龙': 10,
           '疮痂': 11, '早疫': 12, '晚疫': 13, '叶枯': 14, '叶霉': 15, '斑点': 16, '斑枯': 17, '蜘蛛': 18, '曲叶病毒': 19}