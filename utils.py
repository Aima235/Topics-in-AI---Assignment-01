"""Plots, GIF creation, and PDF report for a complete RL submission."""
import os
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import gymnasium as gym

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)


def rolling_mean(values, window=250):
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def plot_training(history, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = history["episode"]
    axes[0].plot(x, rolling_mean(history["reward"]), label="Rolling mean reward (250)")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Q-Learning Convergence on Taxi-v4")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, 100 * rolling_mean(history["success"]), label="Rolling success (250)")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylim(-3, 103)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, parameter, title in zip(axes, ["alpha", "gamma"], ["Learning rate", "Discount factor"]):
        subset = df[df["parameter"] == parameter].sort_values("value")
        ax.plot(subset["value"].to_numpy(), 100 * subset["success_rate"].to_numpy(), "o-", label="Success")
        ax.set_xlabel(title)
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.25)
    fig.suptitle("Hyperparameter Sensitivity (Same 100 Evaluation Seeds)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_success_gif(q_table, output_path, seed_start=50_000, max_trials=200, fps=3):
    """Find a successful greedy rollout; save actual Gymnasium RGB frames."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = gym.make("Taxi-v4", render_mode="rgb_array")
    try:
        for trial in range(max_trials):
            state, _ = env.reset(seed=seed_start + trial)
            frames = [env.render()]
            total_reward = 0.0
            for step in range(200):
                action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                frames.append(env.render())
                if terminated or truncated:
                    if terminated:
                        imageio.mimsave(output_path, [np.asarray(frame) for frame in frames], duration=1/fps, loop=0)
                        return {"seed": seed_start + trial, "reward": total_reward, "steps": step + 1,
                                "frames": len(frames)}
                    break
    finally:
        env.close()
    raise RuntimeError("No successful greedy trajectory found in 200 seeds; try more training episodes.")


def create_report(path, config, train_history, results, sensitivity_df, plot_training_path, plot_sensitivity_path, gif_info):
    """Generate a report with REAL run results, capped to four pages."""
    report_path = Path(path)
    stylesheet = getSampleStyleSheet()
    stylesheet.add(ParagraphStyle(name="MainTitleX", parent=stylesheet["Title"], fontSize=18, leading=22,
                                  spaceAfter=12, alignment=TA_CENTER, textColor=colors.HexColor("#12304a")))
    stylesheet.add(ParagraphStyle(name="HeadingX", parent=stylesheet["Heading2"], fontSize=11,
                                  spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#14436a")))
    stylesheet.add(ParagraphStyle(name="BodyX", parent=stylesheet["BodyText"], fontSize=9, leading=13,
                                  spaceAfter=5))
    stylesheet.add(ParagraphStyle(name="SmallX", parent=stylesheet["BodyText"], fontSize=8, leading=10))
    story = []

    def p(s, style="BodyX"):
        story.append(Paragraph(str(s), stylesheet[style]))

    def h(s):
        p(s, "HeadingX")

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eff6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17344a")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d4dce5")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)
        story.append(Spacer(1, 9))

    p("Custom Reinforcement Learning Agent", "MainTitleX")
    p("Topics in AI | Assignment I (CLO-I) | Taxi-v4 using Q-Learning", "SmallX")
    h("1. Problem Description")
    p("Taxi-v4 is a discrete 5 x 5 navigation problem. The taxi must travel to one of four passenger locations, "
      "pick up the passenger, and deliver them to the indicated destination. "
      "The observation is an encoded integer with 500 possible states: taxi row (5), taxi column (5), "
      "passenger location (5, including inside taxi), and destination (4). There are 6 actions: "
      "south, north, east, west, pickup, and dropoff.")
    p("The environment gives -1 per ordinary time step, -10 for an illegal pickup/dropoff, and +20 for correct "
      "delivery. Success means the episode terminates after delivery; a 200-step limit truncates incomplete episodes.")
    h("2. Methodology")
    p("A tabular Q-learning agent was implemented from scratch. At each step, the epsilon-greedy policy either "
      "explores a uniformly sampled action or exploits a maximum-Q action. The update is:")
    p("<b>Q(s,a) &lt;- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]</b>.")
    p("For a true terminal transition, the bootstrap term is zero. At a time-limit truncation, the learner "
      "still bootstraps, but stops the episode. Exploration decreases linearly toward its floor.")
    p("Implementation: <b>agent.py</b> contains training, exploration, and evaluation; <b>utils.py</b> produces "
      "plots, the real RGB-rendered GIF, and this report; <b>main.py</b> runs and exports the experiment. "
      "Only NumPy is used to represent and update the learned Q-values.")
    h("3. Experimental Setup")
    table([
        ["Setting", "Value", "Setting", "Value"],
        ["Environment", "Taxi-v4", "Episodes", str(config.episodes)],
        ["Learning rate (alpha)", f"{config.alpha}", "Discount (gamma)", f"{config.gamma}"],
        ["Start epsilon", f"{config.epsilon_start}", "End epsilon", f"{config.epsilon_end}"],
        ["Decay fraction", f"{config.decay_fraction}", "Random seed", f"{config.seed}"],
        ["Eval episodes", str(results["trained"]["episodes"]), "Q-table shape", "500 x 6"],
    ], widths=[120, 95, 120, 95])
    story.append(PageBreak())

    h("4. Results: Convergence")
    story.append(Image(str(plot_training_path), width=16.4*cm, height=11.5*cm))
    early_window = min(250, len(train_history["reward"]))
    first_avg = float(np.mean(train_history["reward"][:early_window]))
    final_avg = float(np.mean(train_history["reward"][-early_window:]))
    p(f"Average training reward over the first {early_window} episodes was <b>{first_avg:.2f}</b>; "
      f"over the last {early_window} it was <b>{final_avg:.2f}</b>. "
      "The rolling curves indicate whether learning stabilized and how reliably the taxi completes trips.")
    h("5. Evaluation Against Random Policy")
    table([
        ["Policy", "Mean reward", "Success", "Mean steps"],
        ["Untrained random", f'{results["random"]["mean_reward"]:.2f}',
         f'{100*results["random"]["success_rate"]:.1f}%', f'{results["random"]["mean_steps"]:.1f}'],
        ["Trained greedy", f'{results["trained"]["mean_reward"]:.2f}',
         f'{100*results["trained"]["success_rate"]:.1f}%', f'{results["trained"]["mean_steps"]:.1f}'],
    ], widths=[145, 100, 90, 95])
    p("Both policies use the same 100 evaluation starting seeds. No epsilon-greedy exploration is used for "
      "the trained policy during evaluation. Success is the proportion ending in legal passenger delivery.")
    story.append(PageBreak())

    h("6. Hyperparameter Sensitivity")
    story.append(Image(str(plot_sensitivity_path), width=16.4*cm, height=6.9*cm))
    data_rows = [["Parameter", "Value", "Mean reward", "Success"]]
    for row in sensitivity_df.itertuples(index=False):
        data_rows.append([row.parameter, f"{row.value:g}", f"{row.mean_reward:.2f}", f"{100*row.success_rate:.1f}%"])
    table(data_rows, widths=[125, 75, 120, 110])
    best_row = sensitivity_df.sort_values(["success_rate", "mean_reward"], ascending=False).iloc[0]
    p(f"In the shorter sensitivity runs, the highest ranked setting was <b>{best_row['parameter']}="
      f"{best_row['value']:g}</b> with mean evaluation reward <b>{best_row['mean_reward']:.2f}</b> "
      f"and success rate <b>{100*best_row['success_rate']:.1f}%</b>. "
      "These runs are shorter than the main training run and compare only the listed candidate values.")
    h("7. Discussion and Failure Modes")
    p("Initial episodes can accumulate many negative step rewards and illegal pickup/dropoff penalties. "
      "Training improves through repeated state-action updates. Remaining failures can arise from states that "
      "have been visited too rarely during training, too little exploration, or an unfavorable learning rate. "
      "A high success rate alongside shorter routes is evidence of useful learned behavior; the evaluation "
      "scores and plots above are the measured results rather than assumed outcomes.")
    h("8. Conclusion")
    p(f"This implementation learned a tabular policy for Taxi-v4. Across {results['trained']['episodes']} "
      f"evaluation episodes, the trained agent achieved <b>{100*results['trained']['success_rate']:.1f}%</b> "
      f"successful deliveries and mean reward <b>{results['trained']['mean_reward']:.2f}</b>, compared "
      f"with <b>{results['random']['mean_reward']:.2f}</b> for the random baseline. "
      f"A successful demonstrated rollout took <b>{gif_info['steps']}</b> steps. "
      "The exported Q-table and GIF enable direct inspection of the learned behavior.")
    h("Reference")
    p("Farama Foundation. Gymnasium Taxi-v4 documentation: "
      "https://gymnasium.farama.org/environments/toy_text/taxi/", "SmallX")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(A4[0]/2, 1.15*cm, f"Topics in AI | Technical Report | Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(report_path), pagesize=A4, topMargin=1.65*cm,
                            bottomMargin=1.85*cm, leftMargin=2.0*cm, rightMargin=2.0*cm)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return str(report_path)
