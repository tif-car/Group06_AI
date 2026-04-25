"""
COSC 4368 Check-in 3 — run this file.
Usage: python main.py
Requires: pip install numpy pillow scipy
"""

import os
import time
from maze_solver import (
    MazeEnvironment,
    DynaQAgent,
    LiveVisualizer,
    run_episode,
    evaluate_agent,
    render_solution,
    trace_path,
    visualize_solve,
)

os.makedirs("outputs", exist_ok=True)


def print_metrics(name, m):
    print(f"\n{name.upper()}:")
    print(f"  1. Success rate:           {m['success_rate'] * 100:.0f}%")
    if m["avg_path_length"] == m["avg_path_length"]:
        print(f"  2. Avg path length:        {m['avg_path_length']:.0f}")
    else:
        print("  2. Avg path length:        N/A")

    if m["avg_turns"] == m["avg_turns"]:
        print(f"  3. Avg turns to solution:  {m['avg_turns']:.0f}")
    else:
        print("  3. Avg turns to solution:  N/A")

    print(f"  4. Death rate:             {m['death_rate']:.4f}")
    print(f"  5. Exploration efficiency: {m['exploration_efficiency']:.3f}")
    print(f"  6. Map completeness:       {m['map_completeness']:.3f}")
    print(f"  7. Avg replan time (ms):   {m['avg_replanning_sec'] * 1000:.2f}")
    print(f"  Solution image: outputs/solution_{name}.png")


def train_on_alpha(agent, n_train=15, max_turns=3000):
    print(f"\n{'=' * 55}")
    print("TRAIN ON ALPHA")
    print(f"{'=' * 55}")

    alpha_env = MazeEnvironment("alpha")
    agent.boot(alpha_env)

    print(f"Training ({n_train} episodes on alpha):")
    for ep in range(n_train):
        t = time.perf_counter()
        s = run_episode(alpha_env, agent, max_turns=max_turns)
        elapsed = time.perf_counter() - t

        print(
            f"  Ep {ep + 1}: goal={s['goal_reached']} "
            f"turns={s['turns']:5d}  deaths={s['deaths']:4d}  "
            f"explored={s['cells_explored']:4d}  {elapsed:.1f}s"
        )

        # Optional early stop if it starts succeeding consistently
        if s["goal_reached"] and ep >= 1:
            break

    return alpha_env


def evaluate_and_render(agent, maze_id, max_turns=3000, n_eval=5):
    print(f"\n{'=' * 55}")
    print(f"EVALUATE ON {maze_id.upper()}")
    print(f"{'=' * 55}")

    metrics = evaluate_agent(
        agent,
        maze_id,
        n_ep=n_eval,
        max_turns=max_turns,
        retain=True,
        verbose=True,
    )

    env = MazeEnvironment(maze_id)
    agent.boot(env)  # load this maze into the same trained agent
    path = trace_path(agent, env, max_turns=max_turns)
    render_solution(env, path, f"outputs/solution_{maze_id}.png", scale=6)

    return metrics


def main():
    import sys
    live = "--live" in sys.argv       # pass --live to show real-time window
    #! DEBUGGING!#####################################
    max_turns = 10000
    n_train_alpha = 5
    #! DEBUGGING!#####################################
    n_eval = 5

    agent = DynaQAgent()

    # 1) Train only on alpha
    train_on_alpha(agent, n_train=n_train_alpha, max_turns=max_turns)

    # 2) Evaluate on alpha (optionally with live vis on first episode)
    vis_alpha = None
    if live:
        alpha_env = MazeEnvironment("alpha")
        agent.boot(alpha_env)
        vis_alpha = LiveVisualizer(alpha_env, title="Alpha -- Live")

    alpha_metrics = evaluate_and_render(
        agent, "alpha", max_turns=max_turns, n_eval=n_eval,
    )

    if vis_alpha:
        vis_alpha.close()

    # 3) Evaluate on beta (zero-shot — no training on beta)
    beta_metrics = evaluate_and_render(
        agent, "beta", max_turns=max_turns, n_eval=n_eval,
    )

    if live:
        visualize_solve("beta", max_turns=max_turns, agent=agent)

    # 4) Evaluate on gamma (extra credit)
    print(f"\n{'=' * 55}")
    print("EVALUATE ON GAMMA (extra credit)")
    print(f"{'=' * 55}")
    try:
        gamma_metrics = evaluate_and_render(
            agent, "gamma", max_turns=max_turns, n_eval=n_eval,
        )
    except Exception as exc:
        print(f"  Gamma failed: {exc}")
        gamma_metrics = None

    results = {"alpha": alpha_metrics, "beta": beta_metrics}
    if gamma_metrics:
        results["gamma"] = gamma_metrics

    print(f"\n{'=' * 55}")
    print("FINAL METRICS")
    print(f"{'=' * 55}")

    for name, m in results.items():
        print_metrics(name, m)




if __name__ == "__main__":
    main()