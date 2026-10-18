"""Run the complete Gymnasium Taxi-v4 assignment from Python or Colab."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless Colab GIF rendering

import numpy as np
import pandas as pd
from agent import TrainConfig, train_q_learning, evaluate_policy
from utils import plot_training, plot_sensitivity, save_success_gif, create_report


def run(episodes=20_000, sensitivity_episodes=5_000, eval_episodes=100, output_dir="outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(episodes=episodes)

    print("\n=== PART 1: Q-LEARNING TRAINING ===")
    q_table, history = train_q_learning(config)
    np.save(out / "q_table.npy", q_table)
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    plot_training(history, out / "training_curves.png")
    print(f"Saved Q-table: {q_table.shape}, nonzero entries: {np.count_nonzero(q_table)}")

    print("\n=== PART 2: EVALUATION ===")
    trained = evaluate_policy(q_table, episodes=eval_episodes)
    random_baseline = evaluate_policy(episodes=eval_episodes, random_policy=True)
    evaluations = {"trained": trained, "random": random_baseline}
    print("Trained agent:", trained)
    print("Random baseline:", random_baseline)

    print("\n=== PART 3: HYPERPARAMETER SENSITIVITY ===")
    rows = []
    for name, candidates in (("alpha", [0.1, 0.4, 0.7]), ("gamma", [0.8, 0.95, 0.99])):
        for value in candidates:
            varied = TrainConfig(episodes=sensitivity_episodes)
            setattr(varied, name, value)
            q_candidate, _ = train_q_learning(varied, verbose=False)
            scores = evaluate_policy(q_candidate, episodes=eval_episodes)
            rows.append({"parameter": name, "value": value, **scores})
            print(f"{name}={value:<4} => success={scores['success_rate']:.1%}; mean reward={scores['mean_reward']:.2f}")
    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(out / "hyperparameter_results.csv", index=False)
    plot_sensitivity(sensitivity_df, out / "hyperparameter_comparison.png")

    print("\n=== PART 4: SUCCESSFUL DEMONSTRATION GIF ===")
    gif_info = save_success_gif(q_table, out / "trained_taxi.gif")
    print("Successful rollout:", gif_info)

    print("\n=== PART 5: TECHNICAL REPORT ===")
    pdf_path = create_report(
        out / "technical_report.pdf", config, history, evaluations, sensitivity_df,
        out / "training_curves.png", out / "hyperparameter_comparison.png", gif_info
    )
    print("Report written:", pdf_path)

    summary = {"config": vars(config), "evaluation": evaluations,
               "successful_gif": gif_info, "files_generated": sorted(p.name for p in out.iterdir())}
    (out / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFINISHED. All files are saved under", str(out.resolve()))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate scratch Q-learning on Taxi-v4")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--sensitivity-episodes", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.episodes, args.sensitivity_episodes, args.eval_episodes, args.output_dir)
