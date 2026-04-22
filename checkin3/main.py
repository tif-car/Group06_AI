"""
COSC 4368 Check-in 3 — run this file.
Usage:  python main.py
Requires: pip install numpy pillow scipy
"""
import os, time
from maze_solver import (MazeEnvironment, DynaQAgent, run_episode,
                          evaluate_agent, render_solution, trace_path)

os.makedirs("outputs", exist_ok=True)

def run_maze(maze_id, n_train=5, max_turns=5000):
    print(f"\n{'='*55}\n{maze_id.upper()}\n{'='*55}")
    env = MazeEnvironment(maze_id)
    agent = DynaQAgent()
    agent.boot(env)

    print(f"Training ({n_train} episodes):")
    for ep in range(n_train):
        t = time.perf_counter()
        s = run_episode(env, agent, max_turns=max_turns)
        print(f"  Ep {ep+1}: goal={s['goal_reached']} "
              f"turns={s['turns']:5d}  deaths={s['deaths']:4d}  "
              f"explored={s['cells_explored']:4d}  "
              f"{time.perf_counter()-t:.1f}s")
        if s['goal_reached'] and ep >= 1:
            break

    print(f"\nEvaluating (5 episodes):")
    m = evaluate_agent(agent, maze_id, n_ep=5, max_turns=max_turns, retain=True)

    # Render solution image
    path = trace_path(agent, env, max_turns=max_turns)
    render_solution(env, path, f"outputs/solution_{maze_id}.png", scale=6)

    return m

results = {}
results['alpha'] = run_maze('alpha', n_train=5,  max_turns=5000)
results['beta']  = run_maze('beta',  n_train=8,  max_turns=5000)
results['gamma'] = run_maze('gamma', n_train=8,  max_turns=5000)

print(f"\n{'='*55}\nFINAL METRICS\n{'='*55}")
for name, m in results.items():
    print(f"\n{name.upper()}:")
    print(f"  1. Success rate:           {m['success_rate']*100:.0f}%")
    print(f"  2. Avg path length:        {m['avg_path_length']:.0f}" if m['avg_path_length']==m['avg_path_length'] else "  2. Avg path length:        N/A")
    print(f"  3. Avg turns to solution:  {m['avg_turns']:.0f}" if m['avg_turns']==m['avg_turns'] else "  3. Avg turns to solution:  N/A")
    print(f"  4. Death rate:             {m['death_rate']:.4f}")
    print(f"  5. Exploration efficiency: {m['exploration_efficiency']:.3f}")
    print(f"  6. Map completeness:       {m['map_completeness']:.3f}")
    print(f"  7. Avg replan time (ms):   {m['avg_replanning_sec']*1000:.2f}")
    print(f"  Solution image: outputs/solution_{name}.png")
