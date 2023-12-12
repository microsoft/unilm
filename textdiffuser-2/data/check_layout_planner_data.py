import json
import random
from PIL import Image, ImageDraw, ImageFont

f = open(f'./layout_planner_data_5k.json')
items = json.load(f)
# print(len(items))

dic = random.sample(items, k=1)
print(dic)
layout = dic[0]['conversations'][1]['value']
print(layout)

blank = Image.new('RGB', (256,256), (0,0,0))
draw = ImageDraw.ImageDraw(blank)
font = ImageFont.truetype('../assets/arial.ttf', 16)

for line in layout.split('\n'):
    line = line.strip()

    if len(line) == 0:
        break

    pred = ' '.join(line.split()[:-1])
    box = line.split()[-1]
    l, t, r, b = [int(i)*2 for i in box.split(',')] # the size of canvas is 256x256
    draw.rectangle([(l, t), (r, b)], outline ="red")
    draw.text((l, t), pred, font=font)

blank.save('test.jpg')
f.close()

print('Visualizations are successfully saved at ./test.jpg')