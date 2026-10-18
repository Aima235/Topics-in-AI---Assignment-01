# Custom Q-Learning Agent on Gymnasium Taxi-v3

**Topics in AI, Assignment I**

**Aima Perveen**

**FA23-BAI-066**

**Environment:** `Taxi-v3`

**Algorithm:** tabular Q-Learning implemented from scratch, using NumPy.
**Execution platform:** Google Colab

## Run in Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Taxi_Q_Learning_Colab.ipynb` using **File > Upload notebook**.
3. Choose **Runtime > Run all**. Internet is required only to install missing Python packages.
4. The notebook creates its source files, trains the model, evaluates the agent, produces figures, saves a real Gymnasium-rendered GIF, and generates a measured technical report.
5. Download the ZIP built at the end, and push the extracted project to a new GitHub repository.

## Run as a Python project
```bash
pip install -r requirements.txt
python main.py --episodes 20000 --sensitivity-episodes 5000 --eval-episodes 100
```

## Generated deliverables
Running the notebook or script creates `outputs/`:

- `q_table.npy`: trained Q(s,a), shape 500 x 6
- `trained_taxi.gif`: **actual rendered successful** agent trajectory
- `training_history.csv`, `hyperparameter_results.csv`: experiment data
- `technical_report.pdf`: automatically written with run-specific measurements (maximum 4 pages)

## The Q-Learning update

`Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))`

For true terminal transitions, the bootstrap term is zero. We do not use pre-trained agents or built-in RL algorithms.

## Taxi-v3 environment

- States: 500 encoded discrete observations (taxi row/col, passenger position, destination).
- Actions: 6 (south/north/east/west/pickup/dropoff).
- Rewards: -1 per ordinary step, -10 for illegal pickup/dropoff, +20 for correct delivery.
- Successful delivery terminates an episode, otherwise a time limit truncates it.



