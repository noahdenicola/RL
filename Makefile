install: 
	pip3 install -Ur requirements.txt

saved:
	if [ -d "saved" ]; then echo "Dir exists"; else mkdir saved; fi

ll_dqn: saved
	python3 LunarLander_DQN.py

cp_dqn: saved
	python3 CartPole_DQN.py

cp_reinforce: saved
	python3 CartPole_REINFORCE.py
