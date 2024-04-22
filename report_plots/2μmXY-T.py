import matplotlib.pyplot as plt
import re
import scienceplots
plt.style.use(['science'])
data = """

Frame 1: x=1826.7744140625, y=177.51602172851562
Frame 2: x=1826.3160400390625, y=176.9441375732422
Frame 3: x=1826.787841796875, y=177.9434814453125
Frame 4: x=1827.029541015625, y=178.27099609375
Frame 5: x=1826.747314453125, y=177.7241668701172
Frame 6: x=1826.642333984375, y=177.40328979492188
Frame 7: x=1825.34765625, y=177.6145477294922
Frame 8: x=1824.9287109375, y=177.4997100830078
Frame 9: x=1825.24609375, y=177.38685607910156
Frame 10: x=1825.3746337890625, y=177.30160522460938
Frame 11: x=1825.466064453125, y=177.69876098632812
Frame 12: x=1825.3778076171875, y=177.84591674804688
Frame 13: x=1826.7685546875, y=178.46578979492188
Frame 14: x=1827.2109375, y=178.68649291992188
Frame 15: x=1827.66943359375, y=178.8482208251953
Frame 16: x=1827.8056640625, y=178.96026611328125
Frame 17: x=1828.536865234375, y=178.94334411621094
Frame 18: x=1828.65478515625, y=178.78797912597656
Frame 19: x=1828.29052734375, y=178.46682739257812
Frame 20: x=1828.22119140625, y=178.34371948242188
Frame 21: x=1828.62939453125, y=177.22268676757812
Frame 22: x=1828.810791015625, y=176.62266540527344
Frame 23: x=1830.1883544921875, y=176.19073486328125
Frame 24: x=1830.520751953125, y=175.91969299316406
Frame 25: x=1828.4423828125, y=177.11146545410156
Frame 26: x=1827.7344970703125, y=177.62472534179688
Frame 27: x=1827.156005859375, y=178.89620971679688
Frame 28: x=1827.005615234375, y=179.1436767578125
Frame 29: x=1825.776123046875, y=177.84808349609375
Frame 30: x=1825.373779296875, y=177.58299255371094
Frame 31: x=1825.818603515625, y=177.14764404296875
Frame 32: x=1825.993896484375, y=176.98141479492188
Frame 33: x=1826.0087890625, y=176.43116760253906
Frame 34: x=1826.24462890625, y=176.05435180664062
Frame 35: x=1825.789794921875, y=176.66009521484375
Frame 36: x=1825.7119140625, y=176.88125610351562
Frame 37: x=1825.40283203125, y=176.6873016357422
Frame 38: x=1825.318603515625, y=176.58258056640625
Frame 39: x=1825.0244140625, y=177.19334411621094
Frame 40: x=1825.093017578125, y=177.40260314941406
Frame 41: x=1825.986083984375, y=177.70147705078125
Frame 42: x=1826.223388671875, y=177.82620239257812
Frame 43: x=1825.69580078125, y=177.34182739257812
Frame 44: x=1825.39013671875, y=177.20896911621094
Frame 45: x=1825.235595703125, y=177.50015258789062
Frame 46: x=1825.2099609375, y=177.5475311279297
Frame 47: x=1825.375244140625, y=176.86477661132812
Frame 48: x=1825.5040283203125, y=176.604736328125
Frame 49: x=1825.929931640625, y=176.73513793945312
Frame 50: x=1825.9913330078125, y=176.61502075195312
Frame 51: x=1826.6973876953125, y=176.2497100830078
Frame 52: x=1826.986328125, y=176.21710205078125
Frame 53: x=1827.90087890625, y=176.25608825683594
Frame 54: x=1828.10986328125, y=176.1737823486328
Frame 55: x=1828.521484375, y=176.40806579589844
Frame 56: x=1828.48876953125, y=176.61387634277344
Frame 57: x=1827.521240234375, y=177.29319763183594
Frame 58: x=1827.185302734375, y=177.3646240234375
Frame 59: x=1825.8076171875, y=177.13824462890625
Frame 60: x=1825.620361328125, y=176.86001586914062
Frame 61: x=1826.711181640625, y=176.7501220703125
Frame 62: x=1826.98291015625, y=176.68675231933594
Frame 63: x=1826.212890625, y=178.03338623046875
Frame 64: x=1826.151123046875, y=178.29611206054688
Frame 65: x=1826.44873046875, y=178.25099182128906
Frame 66: x=1826.564208984375, y=178.21009826660156
Frame 67: x=1826.285888671875, y=177.4558868408203
Frame 68: x=1826.1937255859375, y=176.95849609375
Frame 69: x=1825.384765625, y=177.01202392578125
Frame 70: x=1825.23876953125, y=177.20257568359375
Frame 71: x=1824.391845703125, y=177.4125213623047
Frame 72: x=1824.11328125, y=177.26254272460938
Frame 73: x=1824.476806640625, y=177.38746643066406
Frame 74: x=1824.382080078125, y=177.2867431640625
Frame 75: x=1825.06640625, y=178.3952178955078
Frame 76: x=1825.35009765625, y=178.72030639648438
Frame 77: x=1824.8349609375, y=177.19976806640625
Frame 78: x=1824.60498046875, y=176.8809356689453
Frame 79: x=1824.800537109375, y=177.51785278320312
Frame 80: x=1825.147705078125, y=177.6815185546875
Frame 81: x=1826.740478515625, y=176.78753662109375
Frame 82: x=1827.1614990234375, y=176.5890350341797
Frame 83: x=1826.08251953125, y=177.4144287109375
Frame 84: x=1825.8992919921875, y=177.73812866210938
Frame 85: x=1825.853515625, y=176.86624145507812
Frame 86: x=1825.9560546875, y=176.75521850585938
Frame 87: x=1828.2427978515625, y=176.0573272705078
Frame 88: x=1828.4169921875, y=175.9839630126953
Frame 89: x=1827.77783203125, y=176.83062744140625
Frame 90: x=1827.519287109375, y=177.24000549316406
Frame 91: x=1826.7884521484375, y=178.5037841796875
Frame 92: x=1826.86474609375, y=178.92715454101562
Frame 93: x=1826.93896484375, y=178.7704315185547
Frame 94: x=1827.094970703125, y=178.66143798828125
Frame 95: x=1826.09814453125, y=178.13818359375
Frame 96: x=1825.93603515625, y=177.77029418945312
Frame 97: x=1825.403564453125, y=177.2686004638672
Frame 98: x=1825.318359375, y=177.04061889648438
Frame 99: x=1825.63330078125, y=176.88636779785156
Frame 100: x=1825.6610107421875, y=176.72738647460938
Frame 101: x=1826.3466796875, y=178.55233764648438
Frame 102: x=1826.573486328125, y=179.01409912109375
Frame 103: x=1828.016845703125, y=179.14968872070312
Frame 104: x=1828.4140625, y=179.24676513671875
Frame 105: x=1827.9627685546875, y=178.522216796875
Frame 106: x=1827.708251953125, y=178.14035034179688
Frame 107: x=1828.142578125, y=178.06155395507812
Frame 108: x=1828.3521728515625, y=177.8439178466797
Frame 109: x=1827.7823486328125, y=177.3136749267578
Frame 110: x=1827.602294921875, y=177.2480010986328
Frame 111: x=1828.109130859375, y=176.34368896484375
Frame 112: x=1828.203125, y=175.99859619140625
Frame 113: x=1826.845703125, y=177.30831909179688
Frame 114: x=1826.319580078125, y=177.67160034179688
Frame 115: x=1826.302490234375, y=177.65850830078125
Frame 116: x=1826.447021484375, y=177.63916015625
Frame 117: x=1826.92919921875, y=177.91571044921875
Frame 118: x=1826.971923828125, y=177.86740112304688
Frame 119: x=1827.518310546875, y=178.65383911132812
Frame 120: x=1827.6256103515625, y=178.95144653320312
Frame 121: x=1827.423095703125, y=178.65594482421875
Frame 122: x=1827.254150390625, y=178.5242919921875
Frame 123: x=1826.5469970703125, y=178.06536865234375
Frame 124: x=1826.270263671875, y=177.96603393554688
Frame 125: x=1826.2576904296875, y=177.82467651367188
Frame 126: x=1826.3026123046875, y=177.8831787109375
Frame 127: x=1826.7574462890625, y=177.56692504882812
Frame 128: x=1826.949951171875, y=177.5616455078125
Frame 129: x=1827.2685546875, y=177.02952575683594
Frame 130: x=1827.697998046875, y=177.0552215576172
Frame 131: x=1827.2313232421875, y=177.26416015625
Frame 132: x=1827.1268310546875, y=177.41336059570312
Frame 133: x=1827.4033203125, y=176.73435974121094
Frame 134: x=1827.732177734375, y=176.462646484375
Frame 135: x=1826.264404296875, y=176.9158935546875
Frame 136: x=1825.786865234375, y=176.8631591796875
Frame 137: x=1825.254638671875, y=176.65057373046875
Frame 138: x=1825.120849609375, y=176.37100219726562
Frame 139: x=1825.01904296875, y=176.95985412597656
Frame 140: x=1825.1260986328125, y=176.98358154296875
Frame 141: x=1825.849609375, y=177.0118408203125
Frame 142: x=1826.044189453125, y=176.81805419921875
Frame 143: x=1826.5469970703125, y=176.6217041015625
Frame 144: x=1826.726806640625, y=176.35235595703125
Frame 145: x=1825.997314453125, y=176.8167724609375
Frame 146: x=1825.93505859375, y=177.187744140625
Frame 147: x=1825.624267578125, y=177.06475830078125
Frame 148: x=1825.515380859375, y=176.86651611328125
Frame 149: x=1824.9619140625, y=176.64743041992188
Frame 150: x=1824.919189453125, y=176.53790283203125
Frame 151: x=1824.89501953125, y=176.89913940429688
Frame 152: x=1825.068359375, y=176.88137817382812
Frame 153: x=1825.51611328125, y=176.86776733398438
Frame 154: x=1825.714599609375, y=176.74301147460938
Frame 155: x=1825.41796875, y=176.6912841796875
Frame 156: x=1825.3851318359375, y=176.70343017578125
Frame 157: x=1825.301025390625, y=177.19288635253906
Frame 158: x=1825.4188232421875, y=177.2793731689453
Frame 159: x=1824.91259765625, y=176.6434326171875
Frame 160: x=1824.7584228515625, y=176.58367919921875
Frame 161: x=1825.1943359375, y=177.24496459960938
Frame 162: x=1825.391357421875, y=177.36045837402344
Frame 163: x=1825.136962890625, y=177.6986083984375
Frame 164: x=1825.267578125, y=177.76681518554688
Frame 165: x=1825.8355712890625, y=177.06295776367188
Frame 166: x=1826.069580078125, y=176.95703125
Frame 167: x=1826.991943359375, y=175.9739532470703
Frame 168: x=1827.453125, y=175.65170288085938
Frame 169: x=1826.666015625, y=176.6569061279297
Frame 170: x=1826.48193359375, y=176.7642822265625
Frame 171: x=1826.4268798828125, y=176.37405395507812
Frame 172: x=1826.44873046875, y=176.3880615234375
Frame 173: x=1825.9478759765625, y=176.85647583007812
Frame 174: x=1825.793701171875, y=176.99630737304688
Frame 175: x=1825.1502685546875, y=177.00729370117188
Frame 176: x=1824.998291015625, y=177.06092834472656
Frame 177: x=1825.414794921875, y=177.0303955078125
Frame 178: x=1825.41650390625, y=177.16238403320312
Frame 179: x=1824.6348876953125, y=177.80307006835938
Frame 180: x=1824.461669921875, y=177.94000244140625
Frame 181: x=1824.9124755859375, y=178.51849365234375
Frame 182: x=1825.251708984375, y=178.6056671142578
Frame 183: x=1825.185302734375, y=177.66033935546875
Frame 184: x=1825.2738037109375, y=177.16558837890625
Frame 185: x=1825.0557861328125, y=176.5561065673828
Frame 186: x=1824.8603515625, y=176.5189208984375
Frame 187: x=1824.789794921875, y=176.6569061279297
Frame 188: x=1824.65673828125, y=176.63563537597656
Frame 189: x=1824.47412109375, y=176.90257263183594
Frame 190: x=1824.4833984375, y=176.77728271484375
Frame 191: x=1824.48193359375, y=177.51364135742188
Frame 192: x=1824.35546875, y=177.63182067871094
Frame 193: x=1824.67333984375, y=176.86740112304688
Frame 194: x=1824.869873046875, y=176.70040893554688
Frame 195: x=1824.29150390625, y=177.00198364257812
Frame 196: x=1824.206787109375, y=176.94764709472656
Frame 197: x=1824.514404296875, y=176.11309814453125
Frame 198: x=1824.543701171875, y=176.02072143554688
Frame 199: x=1824.4130859375, y=176.65284729003906
Frame 200: x=1824.3131103515625, y=176.9754638671875
Frame 201: x=1826.201416015625, y=177.34417724609375
Frame 202: x=1826.44677734375, y=177.243896484375
Frame 203: x=1826.41357421875, y=177.10690307617188
Frame 204: x=1826.3486328125, y=177.05897521972656
Frame 205: x=1825.325439453125, y=177.8959503173828
Frame 206: x=1825.10693359375, y=178.20587158203125
Frame 207: x=1824.876953125, y=177.3277587890625
Frame 208: x=1824.67626953125, y=177.16404724121094
Frame 209: x=1824.609375, y=177.35379028320312
Frame 210: x=1824.7559814453125, y=177.0985565185547
Frame 211: x=1824.837158203125, y=176.56625366210938
Frame 212: x=1824.565185546875, y=176.3171844482422
Frame 213: x=1824.29736328125, y=177.12965393066406
Frame 214: x=1824.3363037109375, y=177.36947631835938
Frame 215: x=1824.61962890625, y=177.14215087890625
Frame 216: x=1824.644775390625, y=176.91439819335938
Frame 217: x=1824.556396484375, y=176.4576416015625
Frame 218: x=1824.48388671875, y=176.16732788085938
Frame 219: x=1825.4443359375, y=176.71849060058594
Frame 220: x=1825.689453125, y=176.75149536132812
Frame 221: x=1825.51904296875, y=176.62094116210938
Frame 222: x=1825.352294921875, y=176.69818115234375
Frame 223: x=1824.6348876953125, y=177.04498291015625
Frame 224: x=1824.482177734375, y=177.12986755371094
Frame 225: x=1824.468994140625, y=176.951416015625
Frame 226: x=1824.6239013671875, y=176.81692504882812
Frame 227: x=1825.333740234375, y=176.90609741210938
Frame 228: x=1825.37939453125, y=176.92105102539062
Frame 229: x=1824.5975341796875, y=176.904296875
Frame 230: x=1824.296142578125, y=176.9764404296875
Frame 231: x=1824.404052734375, y=177.19522094726562
Frame 232: x=1824.35009765625, y=177.3180694580078
Frame 233: x=1826.133056640625, y=177.2093048095703
Frame 234: x=1826.39013671875, y=177.07591247558594
Frame 235: x=1827.2860107421875, y=176.5285186767578
Frame 236: x=1827.647705078125, y=176.36529541015625
Frame 237: x=1826.8759765625, y=175.26248168945312
Frame 238: x=1826.504638671875, y=174.9146728515625
Frame 239: x=1826.312744140625, y=176.9562530517578
Frame 240: x=1826.2890625, y=177.28945922851562

"""

frames = {}
pattern = re.compile(r'Frame (\d+): x=([\d.]+), y=([\d.]+)')

for line in data.split('\n'):
    if line.strip():
        match = pattern.match(line)
        if match:
            frame_num, x, y = map(float, match.groups())
            frames[frame_num] = {'x': x -1826.7744140625, 'y': y -177.51602172851562}

time_values = [((int(frame)) / 30) for frame in frames.keys()]


plt.plot(time_values, [frame['x'] for frame in frames.values()], marker='.', linestyle='-', label='x(t)')
plt.plot(time_values, [frame['y'] for frame in frames.values()], marker='.', linestyle='-', label='y(t)')


plt.legend()
plt.grid(True)
plt.title(r'Trajectory over Time for 2$\mu m$ Particle for 8s of Tracking')
plt.xlabel('Time(sec)')
plt.ylabel(r'X\&Y Coordinate $ \times 10^{-7}m$')
plt.show()
