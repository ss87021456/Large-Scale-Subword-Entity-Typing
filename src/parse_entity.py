import re, sys, pprint, json
import numpy as np


filename = sys.argv[1]
index_name = sys.argv[2]

with open(filename, 'r') as f:
	dataset = f.read().splitlines()[1:]
	
	keys = list()
	value = list()
	for data in dataset:
		temp = re.split(r"[\t]",data)
		keys.append(temp[0])
		value.append(temp[1:])
	print("length of keys {:3d}".format(len(keys)))

	entity = dict.fromkeys(keys,'@')

	print("start append...")
	for i in range(len(keys)):
		if i % 100 == 0:
			print ("step {:3d} / {:3d}".format(i, len(keys)))
		entity[keys[i]] = keys[i] + ',' + entity[keys[i]]
		for j in range(i+1, len(keys), 1):
			if keys[i] in keys[j]:
				entity[keys[j]] = keys[i] + ',' + entity[keys[j]]

	for key in entity:
		entity[key] = entity[key][:-2].split(',')

	print ("start replace...")
	for key in entity:
		for idx, element in enumerate(entity[key]):
			index = keys.index(element)
			entity[key][idx] = value[index][0]

	# if int(index_name) == 1:
        if index_name == None:
		save_name = filename[:-4]+'_index'
	else:
		save_name = filename[:-4]
		print ("change key name...")
		for i in range(len(keys)):
			if entity.has_key(value[i][0]): # check duplicated key
				entity[value[i][0]] = entity[value[i][0]] + entity.pop(keys[i])
			else:
				entity[value[i][0]] = entity.pop(keys[i])

	pprint.pprint(entity)
	# Save
	with open(save_name+'.json', 'w') as fp:
		json.dump(entity, fp, sort_keys=True, indent=4)


