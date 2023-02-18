ll_dqn:
	if [ -d "saved" ]; then echo "Dir exists"; else mkdir saved; fi
	python3 LunarLander_DQN.py

cp_dqn:
	if [ -d "saved" ]; then echo "Dir exists"; else mkdir saved; fi
	python3 CartPole_DQN.py

cp_reinforce:
	if [ -d "saved" ]; then echo "Dir exists"; else mkdir saved; fi
	python3 CartPole_REINFORCE.py
