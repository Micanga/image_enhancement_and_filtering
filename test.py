import os
from time import sleep

# Running the main code and testing
for i in range(1,7):
	print('#####\nTest Case 0'+str(i))

	print('Program Answer:',end='',flush=True)
	sleep(0.5)

	os.system("python3 ./main.py < ./test/"+str(i)+".in")

	print('Correct Answer:',end='',flush=True)
	sleep(0.5)

	os.system("cat test/"+str(i)+".out")

	print()
